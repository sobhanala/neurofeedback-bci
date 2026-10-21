import keyboard

print("Listening...")
print("Press presenter buttons")
print("ESC to quit\n")

while True:

    event = keyboard.read_event()

    print(
        event.event_type,
        event.name
    )

    if event.name == "esc":
        break




    