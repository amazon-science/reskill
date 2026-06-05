"""Action vocabulary for the registered ScienceWorld environment."""

ACTION_VOCABULARY = """
The agent operates in ScienceWorld (text-based science experiment environment). Actions are free-form text commands:

| Action Template | Example | Description |
|----------------|---------|-------------|
| `go to <location>` | go to kitchen | Navigate |
| `pick up <object>` | pick up thermometer | Take an object |
| `put <object> in <container>` | put metal pot in sink | Place object |
| `pour <source> into <dest>` | pour water into beaker | Transfer liquid |
| `activate <object>` | activate stove | Turn on device |
| `use <object> on <target>` | use thermometer on water | Measure/apply |
| `mix <container>` | mix beaker | Combine contents |
| `wait` | wait | Pass time |
| `look around` | look around | Observe environment |
| `inventory` | inventory | Check held items |
| `focus on <object>` | focus on thermometer | Read measurement |

Trigger patterns should match the action verb patterns above.
"""
