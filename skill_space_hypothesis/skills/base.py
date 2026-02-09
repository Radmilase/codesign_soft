class Skill:
    """
    Минимальный интерфейс навыка (MVP).
    Навык задаёт параметризацию контактного примитива через sample_params().
    """
    name: str = "base"

    def sample_params(self, n: int = 16):
        raise NotImplementedError("Implement sample_params in subclass")
