"""Liveness check: a random head-turn challenge that a printed photo can't pass.

A photo is flat: tilting or moving it never changes where the nose sits
relative to the eyes. A real head turning does. So we ask the person to turn
LEFT or RIGHT (chosen at random, so it can't be prepared in advance) and then
look back at the camera.
"""
import random
import time

WAIT_TURN, WAIT_RETURN, PASSED, FAILED = "wait_turn", "wait_return", "passed", "failed"


class LivenessChallenge:
    def __init__(self, yaw_turned=0.30, yaw_center=0.12, timeout=6.0, direction=None):
        self.direction = direction or random.choice(["LEFT", "RIGHT"])
        self.yaw_turned = yaw_turned
        self.yaw_center = yaw_center
        self.timeout = timeout
        self.started = time.time()
        self.state = WAIT_TURN
        self.baseline = None  # the person's natural yaw when the challenge starts

    @property
    def time_left(self) -> float:
        return max(0.0, self.timeout - (time.time() - self.started))

    @property
    def instruction(self) -> str:
        if self.state == WAIT_TURN:
            return f"Turn your head {self.direction}"
        if self.state == WAIT_RETURN:
            return "Now look back at the camera"
        return ""

    def update(self, yaw: float) -> str:
        """Feed the current yaw each frame; returns the new state."""
        if self.state in (PASSED, FAILED):
            return self.state
        if self.time_left == 0:
            self.state = FAILED
            return self.state

        # Measure relative to how the person was facing at the start, because
        # nobody looks at a camera with exactly yaw = 0.
        if self.baseline is None:
            self.baseline = yaw
        yaw = yaw - self.baseline

        # The webcam image is mirrored (like a selfie), so turning to YOUR left
        # moves your nose to the left of the screen -> yaw becomes negative.
        turned = yaw < -self.yaw_turned if self.direction == "LEFT" else yaw > self.yaw_turned

        if self.state == WAIT_TURN and turned:
            self.state = WAIT_RETURN
        elif self.state == WAIT_RETURN and abs(yaw) < self.yaw_center:
            self.state = PASSED
        return self.state
