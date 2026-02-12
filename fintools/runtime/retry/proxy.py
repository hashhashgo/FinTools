from tenacity import retry, stop_after_attempt, wait_exponential, stop_never
import wrapt

class RetryProxy(wrapt.ObjectProxy):
    def __init__(self, wrapped, attempts=-1, wait=wait_exponential(multiplier=0.1, min=0.1, max=2)):
        super().__init__(wrapped)
        self.__stop__ = stop_after_attempt(attempts) if attempts > 0 else stop_never
        self.__wait__ = wait

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            return super().__getattr__(name)
        attr = super().__getattr__(name)
        if callable(attr):
            return retry(stop=self.__stop__, wait=self.__wait__)(attr)
        else: return attr