"""
Placeholder script — replace this with your real logic.
Anything printed to stdout is captured and sent back to the frontend.
"""

import sys
from datetime import datetime, timezone

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "World"

    # Simulate some work (e.g., data processing, file operations, API calls, etc.)
    print(f"Hello, {name}! Script executed at {datetime.now(timezone.utc).isoformat()}")

    # To simulate a failure, uncomment the next two lines:
    # print("Something went wrong!", file=sys.stderr)
    # sys.exit(1)

if __name__ == "__main__":
    main()
