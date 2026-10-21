# presenter_test.py

import keyboard

print("Listening...")
print("Press ESC to quit\n")

def on_event(event):

    if event.event_type == "down":

        print(
            f"KEY: {event.name}"
        )

keyboard.hook(on_event)

keyboard.wait("esc")









