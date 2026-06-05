"""Action vocabulary for the registered ALFWorld environment."""

ACTION_VOCABULARY = """
The agent operates in ALFWorld (text-based household environment). Actions follow these fixed templates:

| Action Template | Example | Description |
|----------------|---------|-------------|
| `go to <receptacle> <N>` | go to countertop 1 | Navigate to a location |
| `take <object> <N> from <receptacle> <N>` | take apple 1 from fridge 1 | Pick up an object |
| `put <object> <N> in/on <receptacle> <N>` | put mug 2 in/on cabinet 3 | Place object (alternative to move) |
| `move <object> <N> to <receptacle> <N>` | move egg 1 to diningtable 1 | Place object at destination |
| `open <receptacle> <N>` | open drawer 3 | Open a container |
| `close <receptacle> <N>` | close fridge 1 | Close a container |
| `examine <object/receptacle> <N>` | examine countertop 2 | Look at something closely |
| `clean <object> <N> with <receptacle> <N>` | clean plate 1 with sinkbasin 1 | Clean (at sinkbasin/bathtubbasin) |
| `heat <object> <N> with <receptacle> <N>` | heat mug 2 with microwave 1 | Heat (at microwave/stoveburner) |
| `cool <object> <N> with <receptacle> <N>` | cool potato 1 with fridge 1 | Cool (at fridge) |
| `use <object> <N>` | use desklamp 1 | Toggle a device |
| `inventory` | inventory | Check held items |
| `look` | look | Look around |
"""
