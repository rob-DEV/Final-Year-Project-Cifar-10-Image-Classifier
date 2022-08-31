
class Model:
    """
    Base class for LR and CNN providing a name for ModelPersister
    """
    def __init__(self, name) -> None:
        self.name = name