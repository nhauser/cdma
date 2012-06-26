from cdmacore import _factory

#here we need most probably a more sophisticated configuration facility
_factory.init("/usr/lib/cdma/plugins")


def open_dataset(path):
    return _factory.open_dataset(path)

#cleanup function should be called when the python interpreter exits
import atexit
atexit.register(_factory.cleanup)
