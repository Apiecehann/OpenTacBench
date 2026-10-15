"""Complete six-phrase Twinkle Twinkle Little Star, in C major."""
TWINKLE = (
    60,60,67,67,69,69,67,
    65,65,64,64,62,62,60,
    67,67,65,65,64,64,62,
    67,67,65,65,64,64,62,
    60,60,67,67,69,69,67,
    65,65,64,64,62,62,60,
)
HOLD_SECONDS = tuple(.40 if i%7 == 6 else .12 for i in range(len(TWINKLE)))
