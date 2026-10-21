"""
Dynamic Difficulty Adjustment (DDA) Algorithms.

Implements multiple strategies for adjusting game difficulty based on user
cognitive performance metrics (reaction time, accuracy/error rate).
"""
from abc import ABC, abstractmethod
import statistics


class DDAAlgorithm(ABC):
    """Abstract base class for all DDA algorithms."""

    def __init__(self, min_difficulty=1, max_difficulty=10, **kwargs):
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty

    @abstractmethod
    def compute_new_difficulty(self, current_difficulty, history):
        """
        Compute the next difficulty level.

        Args:
            current_difficulty: Current difficulty level (int).
            history: List of dicts with keys:
                - is_correct (bool)
                - reaction_time_ms (float)
                - difficulty_at_trial (int)

        Returns:
            New difficulty level (int).
        """
        pass

    def clamp(self, value):
        return max(self.min_difficulty, min(self.max_difficulty, int(round(value))))


class ThresholdDDA(DDAAlgorithm):
    """
    Baseline algorithm: threshold-based with moving average.

    If recent accuracy > easy_threshold  -> increase difficulty
    If recent accuracy < hard_threshold  -> decrease difficulty
    Otherwise                            -> keep current
    """

    def __init__(self, window_size=5, easy_threshold=0.8, hard_threshold=0.4, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold

    def compute_new_difficulty(self, current_difficulty, history):
        if len(history) < self.window_size:
            return current_difficulty

        recent = history[-self.window_size:]
        accuracy = sum(1 for trial in recent if trial["is_correct"]) / len(recent)

        if accuracy >= self.easy_threshold:
            return self.clamp(current_difficulty + 1)
        elif accuracy <= self.hard_threshold:
            return self.clamp(current_difficulty - 1)
        return current_difficulty


class MovingAverageDDA(DDAAlgorithm):
    """
    Uses exponential moving average of accuracy and reaction time
    to smoothly adjust difficulty.
    """

    def __init__(self, alpha=0.3, target_accuracy=0.7, rt_weight=0.3, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.target_accuracy = target_accuracy
        self.rt_weight = rt_weight
        self._ema_accuracy = None
        self._ema_rt = None

    def compute_new_difficulty(self, current_difficulty, history):
        if not history:
            return current_difficulty

        latest = history[-1]
        acc_val = 1.0 if latest["is_correct"] else 0.0
        rt_val = latest.get("reaction_time_ms", 500)

        if self._ema_accuracy is None:
            self._ema_accuracy = acc_val
            self._ema_rt = rt_val
        else:
            self._ema_accuracy = self.alpha * acc_val + (1 - self.alpha) * self._ema_accuracy
            self._ema_rt = self.alpha * rt_val + (1 - self.alpha) * self._ema_rt

        acc_signal = self._ema_accuracy - self.target_accuracy

        if self._ema_rt > 0:
            rt_signal = -((rt_val - self._ema_rt) / max(self._ema_rt, 1)) * self.rt_weight
        else:
            rt_signal = 0

        combined = acc_signal + rt_signal

        if combined > 0.15:
            return self.clamp(current_difficulty + 1)
        elif combined < -0.15:
            return self.clamp(current_difficulty - 1)
        return current_difficulty


class EloDDA(DDAAlgorithm):
    """
    Elo-rating inspired DDA.

    Treats the game difficulty as an opponent with a rating. The user also
    has a rating. After each trial the ratings update and we pick a difficulty
    level that gives the user roughly a target win-probability.
    """

    def __init__(self, k_factor=32, target_win_prob=0.65, **kwargs):
        super().__init__(**kwargs)
        self.k_factor = k_factor
        self.target_win_prob = target_win_prob
        self.user_rating = 1500.0
        self.difficulty_rating = 1500.0

    def _expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def compute_new_difficulty(self, current_difficulty, history):
        if not history:
            return current_difficulty

        latest = history[-1]
        actual = 1.0 if latest["is_correct"] else 0.0
        expected = self._expected_score(self.user_rating, self.difficulty_rating)

        self.user_rating += self.k_factor * (actual - expected)
        self.difficulty_rating += self.k_factor * (expected - actual)

        target_difficulty_rating = self.user_rating - 400.0 * statistics.log(
            (1.0 / self.target_win_prob) - 1.0, 10
        ) if self.target_win_prob < 1.0 else self.user_rating

        rating_range = self.max_difficulty - self.min_difficulty
        normalized = (target_difficulty_rating - 1000) / 1000
        new_diff = self.min_difficulty + normalized * rating_range

        return self.clamp(new_diff)


class RuleBasedDDA(DDAAlgorithm):
    """
    Rule-based system using multiple behavioral signals.

    Rules:
    1. Streak of correct answers -> increase
    2. Streak of incorrect answers -> decrease
    3. Very fast correct answers -> increase faster
    4. Very slow but correct -> maintain
    5. Combination rules for flow state maintenance
    """

    def __init__(self, streak_threshold=3, fast_rt_ms=300, slow_rt_ms=2000, **kwargs):
        super().__init__(**kwargs)
        self.streak_threshold = streak_threshold
        self.fast_rt_ms = fast_rt_ms
        self.slow_rt_ms = slow_rt_ms

    def compute_new_difficulty(self, current_difficulty, history):
        if len(history) < self.streak_threshold:
            return current_difficulty

        recent = history[-self.streak_threshold:]

        all_correct = all(t["is_correct"] for t in recent)
        all_wrong = all(not t["is_correct"] for t in recent)
        avg_rt = statistics.mean(t.get("reaction_time_ms", 500) for t in recent)

        if all_correct and avg_rt < self.fast_rt_ms:
            return self.clamp(current_difficulty + 2)
        elif all_correct:
            return self.clamp(current_difficulty + 1)
        elif all_wrong:
            return self.clamp(current_difficulty - 2)
        elif avg_rt > self.slow_rt_ms:
            return self.clamp(current_difficulty - 1)

        return current_difficulty


class PerformanceZoneDDA(DDAAlgorithm):
    """
    Flow-state targeting DDA using performance zones.

    Defines zones: Boredom, Flow, Frustration based on accuracy and reaction time.
    Adjusts difficulty to keep user in the Flow zone.
    """

    def __init__(self, window_size=8, flow_acc_low=0.55, flow_acc_high=0.85,
                 flow_rt_low=250, flow_rt_high=1500, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.flow_acc_low = flow_acc_low
        self.flow_acc_high = flow_acc_high
        self.flow_rt_low = flow_rt_low
        self.flow_rt_high = flow_rt_high

    def compute_new_difficulty(self, current_difficulty, history):
        if len(history) < 3:
            return current_difficulty

        recent = history[-self.window_size:]
        accuracy = sum(1 for t in recent if t["is_correct"]) / len(recent)
        avg_rt = statistics.mean(t.get("reaction_time_ms", 500) for t in recent)

        in_flow_acc = self.flow_acc_low <= accuracy <= self.flow_acc_high
        in_flow_rt = self.flow_rt_low <= avg_rt <= self.flow_rt_high

        if in_flow_acc and in_flow_rt:
            return current_difficulty

        adjustment = 0

        if accuracy > self.flow_acc_high:
            adjustment += 1
        elif accuracy < self.flow_acc_low:
            adjustment -= 1

        if avg_rt < self.flow_rt_low:
            adjustment += 1
        elif avg_rt > self.flow_rt_high:
            adjustment -= 1

        return self.clamp(current_difficulty + adjustment)




class EEGEnhancedDDA(DDAAlgorithm):
    """
    Hybrid DDA combining brain activity (EEG) with performance metrics.
    
    Integrates three signals:
    - Performance accuracy (40% weight)
    - Reaction time (30% weight)
    - EEG cognitive load (30% weight)
    
    Special rules:
    - High cognitive load (>0.75) → decrease difficulty even if accuracy is good
    - Low engagement (cognitive_load < 0.3 AND attention > 0.7) → increase difficulty
    """

    def __init__(self, alpha=0.3, target_accuracy=0.7, 
                 accuracy_weight=0.4, rt_weight=0.3, brain_weight=0.3,
                 high_load_threshold=0.75, low_engagement_load=0.3, 
                 high_attention_threshold=0.7, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.target_accuracy = target_accuracy
        self.accuracy_weight = accuracy_weight
        self.rt_weight = rt_weight
        self.brain_weight = brain_weight
        self.high_load_threshold = high_load_threshold
        self.low_engagement_load = low_engagement_load
        self.high_attention_threshold = high_attention_threshold
        
        # EMA tracking
        self._ema_accuracy = None
        self._ema_rt = None
        self._ema_cognitive_load = None
        self._ema_attention = None

    def compute_new_difficulty(self, current_difficulty, history):
        if not history:
            return current_difficulty

        latest = history[-1]
        acc_val = 1.0 if latest["is_correct"] else 0.0
        rt_val = latest.get("reaction_time_ms", 500)
        cognitive_load = latest.get("eeg_cognitive_load", 0.5)
        attention = latest.get("eeg_attention", 0.5)

        # Initialize EMAs on first trial
        if self._ema_accuracy is None:
            self._ema_accuracy = acc_val
            self._ema_rt = rt_val
            self._ema_cognitive_load = cognitive_load
            self._ema_attention = attention
            return current_difficulty

        # Update EMAs
        self._ema_accuracy = self.alpha * acc_val + (1 - self.alpha) * self._ema_accuracy
        self._ema_rt = self.alpha * rt_val + (1 - self.alpha) * self._ema_rt
        self._ema_cognitive_load = self.alpha * cognitive_load + (1 - self.alpha) * self._ema_cognitive_load
        self._ema_attention = self.alpha * attention + (1 - self.alpha) * self._ema_attention

        # Special rule: high cognitive load → decrease difficulty
        if self._ema_cognitive_load > self.high_load_threshold:
            return self.clamp(current_difficulty - 1)

        # Special rule: low engagement (low load + high attention) → increase difficulty
        if (self._ema_cognitive_load < self.low_engagement_load and 
            self._ema_attention > self.high_attention_threshold):
            return self.clamp(current_difficulty + 1)

        # Compute weighted signals
        # 1. Performance accuracy signal
        performance_score = (self._ema_accuracy - self.target_accuracy) * self.accuracy_weight

        # 2. Reaction time signal (faster than average = positive)
        if self._ema_rt > 0:
            rt_score = ((self._ema_rt - rt_val) / self._ema_rt) * self.rt_weight
        else:
            rt_score = 0

        # 3. Brain signal (optimal load = 0.5, attention bonus)
        brain_score = ((0.75 - self._ema_cognitive_load) * 0.7 + 
                      (self._ema_attention - 0.6) * 0.3) * self.brain_weight

        # Combine all signals
        combined_signal = performance_score + rt_score + brain_score

        # Apply threshold for adjustment
        if combined_signal > 0.15:
            return self.clamp(current_difficulty + 1)
        elif combined_signal < -0.15:
            return self.clamp(current_difficulty - 1)
        
        return current_difficulty


# --- Factory ---

DDA_ALGORITHMS = {
    "threshold": ThresholdDDA,
    "moving_average": MovingAverageDDA,
    "elo": EloDDA,
    "rule_based": RuleBasedDDA,
    "performance_zone": PerformanceZoneDDA,
    "eeg_enhanced": EEGEnhancedDDA,
}


def get_dda_algorithm(name, **kwargs):
    """Factory function to get a DDA algorithm by name."""
    cls = DDA_ALGORITHMS.get(name)
    if cls is None:
        raise ValueError(f"Unknown DDA algorithm: {name}. Available: {list(DDA_ALGORITHMS.keys())}")
    return cls(**kwargs)
