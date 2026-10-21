from pywinusb import hid

class PresenterOverride:

    def __init__(self):

        self.override = None

        self.device = None

        devices = hid.find_all_hid_devices()

        for d in devices:

            if d.product_name == "RF PRESENT":

                self.device = d
                break

        if self.device is None:

            print("Presenter not found")
            return

        self.device.open()

        self.device.set_raw_data_handler(
            self._handler
        )

        #print("Presenter connected")

    def _handler(self, data):

        report = data[:3]

        # Volume Up
        if report == [1, 233, 0]:

            self.override = "UP"

        # Volume Down
        elif report == [1, 234, 0]:

            self.override = "LEFT"

    def tab_pressed(self):

        import keyboard

        if keyboard.is_pressed("tab"):

            self.override = "RIGHT"

    def consume_override(self):

        self.tab_pressed()

        value = self.override

        self.override = None

        return value

    def shutdown(self):

        if self.device is not None:

            self.device.close()