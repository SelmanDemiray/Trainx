"""
Class mappings for various datasets used in the application.
These mappings will be used to display meaningful class names during inference.
"""

from typing import Dict, Mapping, Any

# --------------------------
# Core mappings (immutable)
# --------------------------

# MNIST dataset classes (digits 0-9)
MNIST_CLASSES: Dict[int, str] = {i: f"Digit {i}" for i in range(10)}

# EMNIST Letters dataset classes (common labelings differ across sources).
# Two common conventions:
#  - labels 1..26 => 'a'..'z'  (index 0 unused or special)
#  - labels 0..25 => 'a'..'z'  (0-based)
#
# Here we create a mapping that documents the most common convention:
# map 1..26 to 'a'..'z' and set 0 to an explicit placeholder.
EMNIST_LETTERS_CLASSES: Dict[int, str] = {i: f"Letter {chr(i + 96)}" for i in range(1, 27)}
EMNIST_LETTERS_CLASSES[0] = "Letter <unknown_or_offset_0>"  # set intentionally - see note above

# If your EMNIST labeling uses 0..25 -> 'a'..'z', you can build the alternative mapping:
EMNIST_LETTERS_0TO25: Dict[int, str] = {i: f"Letter {chr(i + 97)}" for i in range(26)}

# EMNIST Digits dataset classes (digits 0-9)
EMNIST_DIGITS_CLASSES: Dict[int, str] = {i: f"Digit {i}" for i in range(10)}

# EMNIST Balanced dataset classes (digits 0-9 and uppercase letters A-Z)
EMNIST_BALANCED_CLASSES: Dict[int, str] = {}
EMNIST_BALANCED_CLASSES.update({i: f"Digit {i}" for i in range(10)})
EMNIST_BALANCED_CLASSES.update({i + 10: f"Letter {chr(i + 65)}" for i in range(26)})

# EMNIST ByClass dataset classes (digits 0-9, uppercase A-Z, lowercase a-z)
EMNIST_BYCLASS_CLASSES: Dict[int, str] = {}
EMNIST_BYCLASS_CLASSES.update({i: f"Digit {i}" for i in range(10)})
EMNIST_BYCLASS_CLASSES.update({i + 10: f"Letter {chr(i + 65)} (uppercase)" for i in range(26)})
EMNIST_BYCLASS_CLASSES.update({i + 36: f"Letter {chr(i + 97)} (lowercase)" for i in range(26)})

# EMNIST ByMerge dataset classes (digits 0-9 + case-insensitive letters)
EMNIST_BYMERGE_CLASSES: Dict[int, str] = {}
EMNIST_BYMERGE_CLASSES.update({i: f"Digit {i}" for i in range(10)})
EMNIST_BYMERGE_CLASSES.update({i + 10: f"Letter {chr(i + 65)}/{chr(i + 97)}" for i in range(26)})

# CIFAR-10 dataset classes
CIFAR10_CLASSES: Dict[int, str] = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
}

# CIFAR-100 coarse classes (20 superclasses)
CIFAR100_COARSE_CLASSES: Dict[int, str] = {
    0: "Aquatic mammals",
    1: "Fish",
    2: "Flowers",
    3: "Food containers",
    4: "Fruit and vegetables",
    5: "Household electrical devices",
    6: "Household furniture",
    7: "Insects",
    8: "Large carnivores",
    9: "Large man-made outdoor things",
    10: "Large natural outdoor scenes",
    11: "Large omnivores and herbivores",
    12: "Medium-sized mammals",
    13: "Non-insect invertebrates",
    14: "People",
    15: "Reptiles",
    16: "Small mammals",
    17: "Trees",
    18: "Vehicles 1",
    19: "Vehicles 2",
}

# CIFAR-100 fine classes (all 100 classes)
CIFAR100_FINE_CLASSES: Dict[int, str] = {
    0: "Apple", 1: "Aquarium fish", 2: "Baby", 3: "Bear", 4: "Beaver",
    5: "Bed", 6: "Bee", 7: "Beetle", 8: "Bicycle", 9: "Bottle",
    10: "Bowl", 11: "Boy", 12: "Bridge", 13: "Bus", 14: "Butterfly",
    15: "Camel", 16: "Can", 17: "Castle", 18: "Caterpillar", 19: "Cattle",
    20: "Chair", 21: "Chimpanzee", 22: "Clock", 23: "Cloud", 24: "Cockroach",
    25: "Couch", 26: "Crab", 27: "Crocodile", 28: "Cup", 29: "Dinosaur",
    30: "Dolphin", 31: "Elephant", 32: "Flatfish", 33: "Forest", 34: "Fox",
    35: "Girl", 36: "Hamster", 37: "House", 38: "Kangaroo", 39: "Keyboard",
    40: "Lamp", 41: "Lawn mower", 42: "Leopard", 43: "Lion", 44: "Lizard",
    45: "Lobster", 46: "Man", 47: "Maple tree", 48: "Motorcycle", 49: "Mountain",
    50: "Mouse", 51: "Mushroom", 52: "Oak tree", 53: "Orange", 54: "Orchid",
    55: "Otter", 56: "Palm tree", 57: "Pear", 58: "Pickup truck", 59: "Pine tree",
    60: "Plain", 61: "Plate", 62: "Poppy", 63: "Porcupine", 64: "Possum",
    65: "Rabbit", 66: "Raccoon", 67: "Ray", 68: "Road", 69: "Rocket",
    70: "Rose", 71: "Sea", 72: "Seal", 73: "Shark", 74: "Shrew",
    75: "Skunk", 76: "Skyscraper", 77: "Snail", 78: "Snake", 79: "Spider",
    80: "Squirrel", 81: "Streetcar", 82: "Sunflower", 83: "Sweet pepper", 84: "Table",
    85: "Tank", 86: "Telephone", 87: "Television", 88: "Tiger", 89: "Tractor",
    90: "Train", 91: "Trout", 92: "Tulip", 93: "Turtle", 94: "Wardrobe",
    95: "Whale", 96: "Willow tree", 97: "Wolf", 98: "Woman", 99: "Worm",
}

# --------------------------
# Utility / access functions
# --------------------------

_SUPPORTED_DATASETS = {
    "mnist",
    "emnist-letters",
    "emnist-letters-0to25",  # explicitly the alternate mapping
    "emnist-digits",
    "emnist-balanced",
    "emnist-byclass",
    "emnist-bymerge",
    "cifar10",
    "cifar100",
    "cifar100-coarse",
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")

def get_class_mapping(dataset_name: str) -> Mapping[int, str]:
    """
    Return a copy of the appropriate class mapping based on dataset name.

    Notes:
    - dataset_name matching is case-insensitive and accepts '_' or '-' as separators.
    - For 'emnist-letters' there are two common label conventions. The default mapping
      maps 1..26 -> 'a'..'z' and sets 0 -> '<unknown_or_offset_0>'. If you want
      a 0..25 -> 'a'..'z' mapping, use 'emnist-letters-0to25'.
    - The returned mapping is a shallow copy to avoid accidental mutation of module-level data.
    """
    key = _normalize_name(dataset_name)
    if key == "mnist":
        return dict(MNIST_CLASSES)
    elif key == "emnist-letters":
        return dict(EMNIST_LETTERS_CLASSES)
    elif key == "emnist-letters-0to25":
        return dict(EMNIST_LETTERS_0TO25)
    elif key == "emnist-digits":
        return dict(EMNIST_DIGITS_CLASSES)
    elif key == "emnist-balanced":
        return dict(EMNIST_BALANCED_CLASSES)
    elif key == "emnist-byclass":
        return dict(EMNIST_BYCLASS_CLASSES)
    elif key == "emnist-bymerge":
        return dict(EMNIST_BYMERGE_CLASSES)
    elif key == "cifar10":
        return dict(CIFAR10_CLASSES)
    elif key == "cifar100":
        return dict(CIFAR100_FINE_CLASSES)
    elif key == "cifar100-coarse":
        return dict(CIFAR100_COARSE_CLASSES)
    else:
        # Unknown: return empty mapping (keeps prior behavior). Could raise instead.
        return {}


def get_class_name(dataset_name: str, class_idx: Any, default: str | None = None) -> str:
    """
    Get the class name for a given dataset and class index.

    - dataset_name: name or alias of dataset (case-insensitive).
    - class_idx: integer index (or value convertible to int).
    - default: if provided, returned when index not found; otherwise falls back to "Class {class_idx}".
    """
    mapping = get_class_mapping(dataset_name)
    try:
        idx = int(class_idx)
    except (TypeError, ValueError):
        # non-integer class index, return as-is or formatted
        return str(class_idx)

    if idx in mapping:
        return mapping[idx]
    if default is not None:
        return default
    return f"Class {idx}"

# --------------------------
# Small example / quick tests
# --------------------------
if __name__ == "__main__":
    # Quick sanity checks
    print("MNIST 7 ->", get_class_name("mnist", 7))
    print("CIFAR10 3 ->", get_class_name("CIFAR10", 3))
    print("EMNIST-letters (1..26) 1 ->", get_class_name("emnist-letters", 1))
    print("EMNIST-letters-0to25 0 ->", get_class_name("emnist-letters-0to25", 0))
    print("Unknown dataset ->", get_class_name("my-dataset", 5))
