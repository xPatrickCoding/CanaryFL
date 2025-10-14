

attack_registry = {
}

def register_attack(name):
    def wrapper(cls):
        attack_registry[name] = cls
        return cls
    return wrapper

def get_attack(name):
    return attack_registry.get(name)
