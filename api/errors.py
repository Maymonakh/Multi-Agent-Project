class ApiError(Exception):
    """Domain error with an HTTP status. Raised by services, converted to JSON by the app."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
