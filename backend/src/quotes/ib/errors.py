class IBClientError(Exception):
    """Exception raised for IBClient's errors."""

    def __init__(self, message):
        self.message = message
    
    def __str__(self)
        return self.message