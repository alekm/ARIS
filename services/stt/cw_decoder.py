#!/usr/bin/env python3
"""
CW (Morse Code) Decoder
Decodes Morse code from audio signals using improved envelope detection and timing analysis.
"""
import numpy as np
import scipy.signal as signal
from typing import Optional, Tuple, List
import logging
from collections import Counter
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# International Morse Code dictionary
MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'", '-.-.--': '!',
    '-..-.': '/', '-.--.': '(', '-.--.-': ')', '.-...': '&', '---...': ':',
    '-.-.-.': ';', '-...-': '=', '.-.-.': '+', '-....-': '-', '..--.-': '_',
    '.-..-.': '"', '...-..-': '$', '.--.-.': '@', '...---...': 'SOS'
}


class CWDecoder:
    """
    Improved CW decoder with adaptive thresholding and histogram-based timing analysis.
    
    Algorithm:
    1. Bandpass filter to isolate CW tone
    2. Envelope detection with improved smoothing
    3. Adaptive threshold with hysteresis
    4. Histogram-based timing analysis to find dot/dash thresholds
    5. Convert to Morse code symbols
    6. Decode to text
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 tone_freq: float = 600.0,
                 wpm: float = 20.0,
                 threshold_ratio: float = 0.4):
        # Validate inputs
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {sample_rate} (must be > 0)")
        if tone_freq <= 0 or tone_freq >= sample_rate / 2:
            raise ValueError(f"Invalid tone_freq: {tone_freq}Hz (must be > 0 and < Nyquist {sample_rate/2}Hz)")
        if wpm <= 0 or wpm > 100:
            raise ValueError(f"Invalid wpm: {wpm} (must be > 0 and <= 100)")
        
        self.sample_rate = sample_rate
        self.tone_freq = tone_freq
        self.wpm = wpm
        self.threshold_ratio = threshold_ratio
        
        # Calculate timing based on WPM (will be refined by histogram analysis)
        self.dot_duration = 60.0 / (50.0 * wpm)  # seconds
        self.dash_duration = 3 * self.dot_duration
        self.element_space = self.dot_duration
        self.char_space = 3 * self.dot_duration
        self.word_space = 7 * self.dot_duration
        
        # Bandpass filter for CW tone
        bandwidth = 200  # Wider bandwidth to capture signal variations
        low = max(1, tone_freq - bandwidth/2)
        high = min(sample_rate/2 - 1, tone_freq + bandwidth/2)
        
        # Validate filter range
        if low >= high:
            raise ValueError(f"Invalid filter range: low={low}Hz >= high={high}Hz (tone_freq={tone_freq}Hz, sample_rate={sample_rate}Hz)")
        
        try:
            self.b, self.a = signal.butter(4, [low, high], btype='band', fs=sample_rate)
        except Exception as e:
            raise ValueError(f"Failed to create bandpass filter: {e} (low={low}Hz, high={high}Hz, sample_rate={sample_rate}Hz)")
        
        # Frequency smoothing: track recent detections to avoid filter churn
        self._detected_freqs = []  # Recent frequency detections
        self._max_freq_history = 5  # Keep last 5 detections
        self._filter_update_threshold = 80.0  # Hz - only update if smoothed freq changes by this much
        
        logger.info(f"CW Decoder initialized: {tone_freq}Hz, {wpm}WPM, threshold={threshold_ratio}, sample_rate={sample_rate}Hz, filter={low:.1f}-{high:.1f}Hz")
    
    def detect_tone_frequency(self, audio: np.ndarray) -> float:
        """Auto-detect the CW tone frequency using FFT peak detection."""
        # Use FFT to find dominant frequency
        # Apply window to reduce spectral leakage
        windowed = audio * np.hanning(len(audio))
        fft = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(audio), 1.0/self.sample_rate)
        magnitude = np.abs(fft)
        
        # Look for peak in CW range (300-1000 Hz)
        cw_range = (freqs >= 300) & (freqs <= 1000)
        if np.any(cw_range):
            # Find the peak, but also check if it's significantly above noise
            magnitude_cw = magnitude[cw_range]
            peak_idx = np.argmax(magnitude_cw)
            peak_freq = float(freqs[cw_range][peak_idx])
            peak_magnitude = float(magnitude_cw[peak_idx])
            
            # Check if peak is significant (at least 3x the median in CW range - more strict)
            median_magnitude = float(np.median(magnitude_cw))
            if peak_magnitude > median_magnitude * 3.0:  # Increased from 2.0
                logger.debug(f"Detected CW tone frequency: {peak_freq:.1f} Hz (magnitude: {peak_magnitude:.1f} vs median: {median_magnitude:.1f})")
                return peak_freq
            else:
                logger.debug(f"Peak at {peak_freq:.1f} Hz not significant enough (magnitude: {peak_magnitude:.1f} vs median: {median_magnitude:.1f}, need 3x)")
        
        return self.tone_freq
    
    def filter_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to isolate CW tone."""
        try:
            if len(audio) == 0:
                logger.warning("Empty audio array, cannot filter")
                return audio
            filtered = signal.filtfilt(self.b, self.a, audio)
            return filtered
        except Exception as e:
            logger.warning(f"Filter error: {e}, using unfiltered audio")
            return audio
    
    def detect_envelope(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect envelope with improved smoothing.
        Uses Hilbert transform + exponential smoothing for better noise rejection.
        """
        # Hilbert transform for analytic signal
        analytic = signal.hilbert(audio)
        envelope = np.abs(analytic)
        
        # Exponential smoothing (better than moving average for tracking)
        # Use a longer window for CW (20ms instead of 10ms)
        alpha = 0.95  # Smoothing factor
        smoothed = np.zeros_like(envelope)
        smoothed[0] = envelope[0]
        for i in range(1, len(envelope)):
            smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * envelope[i]
        
        return smoothed
    
    def detect_on_off_states_adaptive(self, envelope: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Adaptive threshold detection with hysteresis and noise rejection.
        Returns: (states, on_threshold, off_threshold)
        """
        # Estimate noise floor using median (more robust than percentile for noise)
        noise_floor = np.median(envelope)
        noise_std = np.std(envelope)
        
        # Estimate signal level using high percentile
        signal_level = np.percentile(envelope, 95)
        
        # Check signal-to-noise ratio
        snr_db = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))
        logger.debug(f"Signal analysis: noise_floor={noise_floor:.6f}, signal_level={signal_level:.6f}, SNR={snr_db:.1f}dB")
        
        # Require minimum SNR of 4dB to avoid detecting noise (reduced from 6dB for weak beacons)
        if snr_db < 4.0:
            logger.debug(f"SNR too low: {snr_db:.1f}dB, rejecting")
            # Return all-off states
            return np.zeros(len(envelope), dtype=bool), signal_level, noise_floor
        
        # Check dynamic range
        dynamic_range = signal_level - noise_floor
        max_envelope = np.max(envelope)
        
        if dynamic_range < max_envelope * 0.15:  # Need at least 15% dynamic range
            logger.debug(f"Poor signal dynamic range: {dynamic_range:.4f} (max: {max_envelope:.4f})")
            # Return all-off states - likely just noise
            return np.zeros(len(envelope), dtype=bool), signal_level, noise_floor
        
        # More conservative threshold: use noise floor + multiple standard deviations
        # This is more robust to noise spikes
        # For CW, be more aggressive about rejecting noise
        on_threshold = noise_floor + max(
            noise_std * 4.0,  # At least 4 sigma above noise (increased from 3.0)
            dynamic_range * self.threshold_ratio  # Or use ratio of dynamic range
        )
        
        # Hysteresis: lower threshold for "off" but still above noise
        off_threshold = noise_floor + max(
            noise_std * 2.0,  # At least 2 sigma above noise (increased from 1.5)
            dynamic_range * (self.threshold_ratio * 0.6)  # Increased from 0.5
        )
        
        # Ensure thresholds are reasonable
        if on_threshold > max_envelope * 0.9:
            logger.debug(f"Threshold too high: {on_threshold:.6f} vs max {max_envelope:.6f}")
            on_threshold = max_envelope * 0.7  # Use 70% of max as fallback
        
        # Apply hysteresis thresholding with minimum duration requirements
        # Use longer minimum durations to filter out noise spikes
        # For 12kHz: 8ms = 96 samples, 10ms = 120 samples
        states = np.zeros(len(envelope), dtype=bool)
        current_state = False
        state_start = 0
        min_on_samples = int(self.sample_rate * 0.010)  # Minimum 10ms for "on" (increased from 8ms)
        min_off_samples = int(self.sample_rate * 0.010)  # Minimum 10ms for "off" (increased from 8ms)
        
        # Track state changes with debouncing
        for i in range(len(envelope)):
            if current_state:
                # Currently "on" - need to drop below off_threshold to turn off
                if envelope[i] < off_threshold:
                    # Check if we've been "on" long enough (debounce)
                    if i - state_start >= min_on_samples:
                        current_state = False
                        state_start = i
                # If still above threshold, keep state
            else:
                # Currently "off" - need to rise above on_threshold to turn on
                if envelope[i] > on_threshold:
                    # Check if we've been "off" long enough (debounce)
                    if i - state_start >= min_off_samples:
                        current_state = True
                        state_start = i
                # If still below threshold, keep state
            
            states[i] = current_state
        
        # Post-process: remove very short on/off periods (likely noise)
        # This helps clean up the state signal
        cleaned_states = np.copy(states)
        i = 0
        while i < len(cleaned_states):
            if cleaned_states[i]:
                start = i
                while i < len(cleaned_states) and cleaned_states[i]:
                    i += 1
                duration = (i - start) / self.sample_rate
                # Remove on periods shorter than 5ms (likely noise) - increased from 3ms
                if duration < 0.005:
                    cleaned_states[start:i] = False
            else:
                i += 1
        
        # Also remove very short off periods that might be glitches
        i = 0
        while i < len(cleaned_states):
            if not cleaned_states[i]:
                start = i
                while i < len(cleaned_states) and not cleaned_states[i]:
                    i += 1
                duration = (i - start) / self.sample_rate
                # Remove off periods shorter than 2ms (likely glitches)
                if duration < 0.002:
                    # Merge with previous on period
                    if start > 0:
                        cleaned_states[start:i] = True
            else:
                i += 1
        
        return cleaned_states, on_threshold, off_threshold
    
    def analyze_timing_kmeans(self, states: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Analyze on/off durations using K-Means clustering (inspired by morse-audio-decoder).
        This automatically finds dot/dash clusters without needing to know WPM.
        Returns: (dot_duration, dash_threshold, char_space_threshold, word_space_threshold)
        """
        # Collect all on and off durations
        on_durations = []
        off_durations = []
        
        i = 0
        while i < len(states):
            if states[i]:
                start = i
                while i < len(states) and states[i]:
                    i += 1
                duration = (i - start) / self.sample_rate
                if duration > 0.005:  # Ignore very short pulses (< 5ms)
                    on_durations.append(duration)
            else:
                start = i
                while i < len(states) and not states[i]:
                    i += 1
                duration = (i - start) / self.sample_rate
                if duration > 0.005:  # Ignore very short gaps
                    off_durations.append(duration)
            if i == start:  # Prevent infinite loop
                i += 1
        
        if len(on_durations) < 5:
            # Not enough data, use defaults
            return self.dot_duration, self.dash_duration, self.char_space, self.word_space
        
        # Use K-Means to cluster on durations into dots and dashes
        on_durations_array = np.array(on_durations).reshape(-1, 1)
        
        # Try 2 clusters (dots and dashes)
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(on_durations_array)
            centers = sorted(kmeans.cluster_centers_.flatten())
            
            # Shorter cluster = dots, longer cluster = dashes
            dot_duration = float(centers[0])
            dash_center = float(centers[1])
            
            # Validate clusters make sense (dash should be 1.5-5x dot)
            ratio = dash_center / dot_duration if dot_duration > 0 else 0
            if ratio < 1.5 or ratio > 5.0:
                logger.warning(f"Invalid cluster ratio: {ratio:.2f} (dash={dash_center*1000:.1f}ms, dot={dot_duration*1000:.1f}ms), using histogram")
                # Fallback to histogram
                on_durations_sorted = sorted(on_durations)
                # Use median of lower half as dot
                dot_duration = float(np.median(on_durations_sorted[:len(on_durations_sorted)//2]))
                dash_threshold = dot_duration * 2.5
            else:
                # Use actual data distribution to find better threshold
                # Find the "valley" between the two clusters in the sorted durations
                on_durations_sorted = sorted(on_durations)
                
                # Find where the gap is between clusters
                # Look for the largest gap in the sorted list
                max_gap = 0
                gap_idx = 0
                for i in range(len(on_durations_sorted) - 1):
                    gap = on_durations_sorted[i+1] - on_durations_sorted[i]
                    if gap > max_gap:
                        max_gap = gap
                        gap_idx = i
                
                # Use the midpoint of the largest gap as threshold
                if max_gap > dot_duration * 0.5:  # Significant gap found
                    dash_threshold = (on_durations_sorted[gap_idx] + on_durations_sorted[gap_idx+1]) / 2.0
                    logger.debug(f"Using gap-based threshold: {dash_threshold*1000:.1f}ms (gap: {max_gap*1000:.1f}ms)")
                else:
                    # No clear gap, use weighted average
                    dash_threshold = dot_duration + (dash_center - dot_duration) * 0.55
                
                # Ensure minimum separation (at least 1.5x dot)
                if dash_threshold < dot_duration * 1.5:
                    dash_threshold = dot_duration * 1.5
                
                # But don't make it too high (max 2.2x dot for threshold to catch more dashes)
                if dash_threshold > dot_duration * 2.2:
                    dash_threshold = dot_duration * 2.2
            
            logger.debug(f"K-Means clustering: dot cluster={dot_duration*1000:.1f}ms, dash cluster={dash_center*1000:.1f}ms, ratio={ratio:.2f}")
        except Exception as e:
            logger.warning(f"K-Means clustering failed: {e}, using histogram method")
            # Fallback to histogram method
            on_durations_sorted = sorted(on_durations)
            dot_duration = float(np.median(on_durations_sorted[:len(on_durations_sorted)//2]))
            dash_threshold = dot_duration * 2.5
        
        # For off durations, use K-Means if we have enough data
        if len(off_durations) >= 3:
            try:
                off_durations_array = np.array(off_durations).reshape(-1, 1)
                # 3 clusters: element space, character space, word space
                kmeans_off = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans_off.fit(off_durations_array)
                centers_off = sorted(kmeans_off.cluster_centers_.flatten())
                
                # Shortest = element space, middle = char space, longest = word space
                char_space_threshold = (centers_off[0] + centers_off[1]) / 2.0
                word_space_threshold = (centers_off[1] + centers_off[2]) / 2.0
                
                logger.debug(f"Off durations: element={centers_off[0]*1000:.1f}ms, char={centers_off[1]*1000:.1f}ms, word={centers_off[2]*1000:.1f}ms")
            except Exception as e:
                logger.warning(f"K-Means for off durations failed: {e}, using defaults")
                char_space_threshold = dot_duration * 2.5
                word_space_threshold = dot_duration * 5.0
        else:
            # Fallback to defaults based on dot duration
            char_space_threshold = dot_duration * 2.5
            word_space_threshold = dot_duration * 5.0
        
        # Update WPM estimate
        estimated_wpm = 60.0 / (50.0 * dot_duration)
        estimated_wpm = float(max(5, min(50, estimated_wpm)))
        
        logger.debug(f"Timing analysis: dot={dot_duration*1000:.1f}ms, dash_threshold={dash_threshold*1000:.1f}ms, WPM={estimated_wpm:.1f}")
        
        return dot_duration, dash_threshold, char_space_threshold, word_space_threshold
    
    def classify_timing(self, states: np.ndarray, 
                       dot_duration: float, dash_threshold: float,
                       char_space_threshold: float, word_space_threshold: float) -> List[str]:
        """
        Classify on/off durations into dots, dashes, and spaces using learned thresholds.
        """
        if len(states) == 0:
            return []
        
        symbols = []
        i = 0
        prev_i = -1  # Track previous position to prevent infinite loops
        
        while i < len(states):
            # Prevent infinite loop
            if i == prev_i:
                i += 1
                continue
            prev_i = i
            
            if states[i]:
                # On period
                on_start = i
                while i < len(states) and states[i]:
                    i += 1
                on_duration = (i - on_start) / self.sample_rate
                
                # Classify as dot or dash
                if on_duration < dash_threshold:
                    symbols.append('.')
                else:
                    symbols.append('-')
            else:
                # Off period
                off_start = i
                while i < len(states) and not states[i]:
                    i += 1
                off_duration = (i - off_start) / self.sample_rate
                
                # Classify space type
                if off_duration > word_space_threshold:
                    symbols.append(' / ')  # Word space
                elif off_duration > char_space_threshold:
                    symbols.append(' ')  # Character space
                # Element space is implicit (no symbol)
        
        return symbols
    
    def decode_morse(self, morse_symbols: List[str]) -> str:
        """Convert Morse code symbols to text."""
        morse_str = ''.join(morse_symbols)
        
        # Split by word spaces
        words = morse_str.split(' / ')
        decoded_words = []
        
        for word in words:
            if not word.strip():
                continue
            
            # Split word into characters (separated by spaces)
            chars = word.split()
            decoded_chars = []
            
            for char_morse in chars:
                if char_morse in MORSE_CODE:
                    decoded_chars.append(MORSE_CODE[char_morse])
                else:
                    # Unknown pattern
                    decoded_chars.append('?')
                    logger.debug(f"Unknown Morse pattern: {char_morse}")
            
            if decoded_chars:
                decoded_words.append(''.join(decoded_chars))
        
        result = ' '.join(decoded_words)
        return result.strip()
    
    def decode(self, audio: np.ndarray, auto_detect: bool = True) -> Tuple[Optional[str], float]:
        """
        Main decode function with improved algorithm.
        """
        if len(audio) < self.sample_rate * 0.1:  # Need at least 100ms
            return None, 0.0
        
        try:
            # Check audio RMS to see if there's any signal at all
            audio_rms = float(np.sqrt(np.mean(audio**2)))
            if audio_rms < 0.001:  # Very quiet
                logger.debug(f"Audio too quiet: RMS={audio_rms:.6f}")
                return None, 0.0
            
            # Auto-detect tone frequency if enabled
            # Use smoothing to avoid filter churn from noisy detections
            if auto_detect:
                detected_freq = self.detect_tone_frequency(audio)
                
                # Add to history and maintain sliding window
                self._detected_freqs.append(detected_freq)
                if len(self._detected_freqs) > self._max_freq_history:
                    self._detected_freqs.pop(0)
                
                # Calculate smoothed frequency (median of recent detections)
                if len(self._detected_freqs) >= 3:  # Need at least 3 samples for stability
                    smoothed_freq = float(np.median(self._detected_freqs))
                    logger.info(f"Detected CW tone: {detected_freq:.1f} Hz, smoothed: {smoothed_freq:.1f} Hz (filter: {self.tone_freq:.1f} Hz)")
                    
                    # Only update filter if smoothed frequency changed significantly
                    # This prevents filter from jumping around when individual detections are noisy
                    if abs(smoothed_freq - self.tone_freq) > self._filter_update_threshold:
                        # Use wider bandwidth for better signal capture
                        bandwidth = 200
                        low = max(1, smoothed_freq - bandwidth/2)
                        high = min(self.sample_rate/2 - 1, smoothed_freq + bandwidth/2)
                        num_samples = len(self._detected_freqs)
                        self.b, self.a = signal.butter(4, [low, high], btype='band', fs=self.sample_rate)
                        self.tone_freq = smoothed_freq
                        # Clear history after update to start fresh
                        self._detected_freqs = []
                        logger.info(f"Updated filter: {low:.1f}-{high:.1f} Hz (bandwidth: {bandwidth}Hz, smoothed from {num_samples} samples)")
                    else:
                        logger.debug(f"Smoothed frequency ({smoothed_freq:.1f} vs {self.tone_freq:.1f}) change too small, keeping existing filter")
                else:
                    # Not enough samples yet, just log
                    logger.info(f"Detected CW tone: {detected_freq:.1f} Hz (collecting samples: {len(self._detected_freqs)}/{self._max_freq_history})")
            
            # Filter audio
            filtered = self.filter_audio(audio)
            filtered_rms = float(np.sqrt(np.mean(filtered**2)))
            logger.debug(f"Filtered audio RMS: {filtered_rms:.6f} (original: {audio_rms:.6f})")
            
            # Detect envelope
            envelope = self.detect_envelope(filtered)
            envelope_max = float(np.max(envelope))
            envelope_mean = float(np.mean(envelope))
            logger.debug(f"Envelope: max={envelope_max:.6f}, mean={envelope_mean:.6f}")
            
            # Detect on/off states with adaptive thresholding
            states, on_thresh, off_thresh = self.detect_on_off_states_adaptive(envelope)
            logger.info(f"Thresholds: on={on_thresh:.6f}, off={off_thresh:.6f}, envelope_max={envelope_max:.6f}")
            
            # Check if we have any signal
            if not np.any(states):
                logger.debug("No signal detected (all states are off)")
                return None, 0.0
            
            # Check signal quality: need reasonable on/off ratio
            on_ratio = np.sum(states) / len(states)
            logger.info(f"Signal on/off ratio: {on_ratio:.2%} ({np.sum(states)}/{len(states)} samples)")
            
            # For CW, typical on/off ratio depends on message content
            # Beacons with long pauses might have low ratios (3-10%)
            # Fast CW might have higher ratios (20-40%)
            # Too high (>80%) suggests noise or constant tone
            # Too low (<2%) suggests no signal or very weak
            if on_ratio < 0.02:
                logger.debug(f"Too little signal: on_ratio={on_ratio:.2%} (likely noise or no signal)")
                return None, 0.0
            if on_ratio > 0.80:
                logger.debug(f"Too much signal: on_ratio={on_ratio:.2%} (likely constant tone or noise)")
                return None, 0.0
            
            # For very low ratios (2-5%), require stronger signal quality
            if on_ratio < 0.05:
                # Check if we have enough actual signal periods (not just noise)
                # Count distinct on periods
                on_periods = 0
                was_on = False
                for state in states:
                    if state and not was_on:
                        on_periods += 1
                    was_on = state
                
                # Need at least 5 distinct on periods for a valid signal
                if on_periods < 5:
                    logger.debug(f"Too few signal periods: {on_periods} periods with {on_ratio:.2%} on ratio")
                    return None, 0.0
                
                logger.info(f"Low on ratio ({on_ratio:.2%}) but {on_periods} distinct periods - likely beacon with pauses")
            
            # Check for too many rapid state changes (noise characteristic)
            state_changes = np.sum(np.diff(states.astype(int)) != 0)
            change_rate = state_changes / len(states)
            if change_rate > 0.1:  # More than 10% of samples are state changes
                logger.debug(f"Too many state changes: {state_changes} changes ({change_rate:.1%} rate) - likely noise")
                return None, 0.0
            
            # Analyze timing using K-Means clustering (more robust than histograms)
            dot_dur, dash_thresh, char_space_thresh, word_space_thresh = self.analyze_timing_kmeans(states)
            
            logger.info(f"Timing analysis: dot={dot_dur*1000:.1f}ms, dash_threshold={dash_thresh*1000:.1f}ms, "
                       f"char_space={char_space_thresh*1000:.1f}ms, word_space={word_space_thresh*1000:.1f}ms")
            
            # Validate timing thresholds are reasonable
            # Allow faster speeds: 5ms (60 WPM) to 500ms (very slow)
            if dot_dur < 0.005 or dot_dur > 0.5:
                logger.warning(f"Invalid dot duration: {dot_dur*1000:.1f}ms, rejecting")
                return None, 0.0
            
            if dash_thresh < dot_dur * 1.5:  # Dash should be at least 1.5x dot
                logger.warning(f"Dash threshold too close to dot: {dash_thresh*1000:.1f}ms vs {dot_dur*1000:.1f}ms, adjusting")
                dash_thresh = dot_dur * 2.0  # Fix it
            
            # Classify timing into Morse symbols
            morse_symbols = self.classify_timing(states, dot_dur, dash_thresh, char_space_thresh, word_space_thresh)
            
            if not morse_symbols:
                logger.debug("No Morse symbols generated")
                return None, 0.0
            
            # Log the raw Morse pattern for debugging
            morse_pattern = ''.join(morse_symbols)
            logger.info(f"Raw Morse pattern: {morse_pattern[:100]}... ({len(morse_symbols)} symbols)")
            
            # Validate we have a reasonable number of symbols
            if len(morse_symbols) < 3:
                logger.debug(f"Too few symbols: {len(morse_symbols)}")
                return None, 0.0
            
            # Check for suspicious patterns (too many consecutive dots/dashes suggests timing error)
            # Count max consecutive same symbol within a character (between spaces)
            max_consecutive = 0
            current_consecutive = 0
            last_symbol = None
            for sym in morse_symbols:
                if sym in ['.', '-']:
                    if sym == last_symbol:
                        current_consecutive += 1
                    else:
                        current_consecutive = 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                    last_symbol = sym
                elif sym in [' ', ' / ']:
                    # Reset on space (new character)
                    current_consecutive = 0
                    last_symbol = None
            
            # If we have more than 8 consecutive dots or dashes in a character, likely a timing error
            # (Longest valid Morse character is 6 elements: e.g., .----. for "1")
            # Allow 8 to account for timing variation and some noise
            if max_consecutive > 8:
                logger.debug(f"Too many consecutive symbols in character: {max_consecutive} (likely timing error or noise)")
                return None, 0.0
            
            # Decode to text
            decoded_text = self.decode_morse(morse_symbols)
            
            if not decoded_text or decoded_text.strip() == '':
                return None, 0.0
            
            # Count question marks (unknown patterns)
            question_mark_ratio = decoded_text.count('?') / len(decoded_text) if decoded_text else 1.0
            
            # Reject if too many unknown patterns (likely bad decode)
            # For beacons, allow up to 40% unknown (they might have unusual patterns)
            if question_mark_ratio > 0.4:
                logger.debug(f"Rejecting decode with too many unknowns: '{decoded_text}' ({question_mark_ratio:.1%} unknown)")
                return None, 0.0
            
            # Reject if decode is too short or mostly punctuation
            if len(decoded_text.strip()) < 3:
                logger.debug(f"Rejecting decode: too short '{decoded_text}'")
                return None, 0.0
            
            # Reject if decode is mostly question marks and single characters (garbled)
            # Count valid characters (letters, numbers, spaces)
            valid_chars = sum(1 for c in decoded_text if c.isalnum() or c.isspace())
            valid_ratio = valid_chars / len(decoded_text) if decoded_text else 0
            if valid_ratio < 0.5:
                logger.debug(f"Rejecting decode: too few valid characters '{decoded_text}' ({valid_ratio:.1%} valid)")
                return None, 0.0
            
            # Calculate confidence based on signal quality
            signal_power = float(np.mean(envelope[states])) if np.any(states) else 0.0
            noise_power = float(np.mean(envelope[~states])) if np.any(~states) else 0.0
            snr = signal_power / (noise_power + 1e-10)
            
            # Require minimum SNR for any confidence (reduced for weak signals)
            if snr < 1.5:  # Need at least 1.5:1 SNR (reduced from 2:1)
                logger.debug(f"SNR too low for confidence: {snr:.2f}")
                return None, 0.0
            
            confidence = float(min(1.0, (snr - 1.5) / 8.5))  # Scale from 1.5-10 SNR to 0-1 confidence
            
            # Boost confidence if we decoded actual letters/numbers (not just '?')
            if question_mark_ratio < 0.1 and len(decoded_text) > 3:
                confidence = min(1.0, confidence * 1.3)
            elif question_mark_ratio > 0.2:
                # Reduce confidence for many unknowns
                confidence = confidence * 0.6
            
            # Additional validation: reject if confidence is too low
            if confidence < 0.25:
                logger.debug(f"Rejecting decode: low confidence {confidence:.2f} for '{decoded_text}'")
                return None, 0.0
            
            logger.info(f"CW decoded: '{decoded_text}' (confidence: {confidence:.2f}, unknowns: {question_mark_ratio:.1%}, valid: {valid_ratio:.1%})")
            
            return decoded_text, confidence
            
        except Exception as e:
            logger.error(f"CW decoding error: {e}", exc_info=True)
            return None, 0.0
