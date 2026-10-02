class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        self.code = code
        self.message = message
        self.status = status
        super().__init__(code)


def require(condition: bool, code: str, message: str, status: int = 409):
    if not condition:
        raise ServiceError(code, message, status)
