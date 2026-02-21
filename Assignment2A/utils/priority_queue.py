"""List-backed deterministic priority queue."""


class PriorityQueue:
    def __init__(self):
        self._items = []

    def push(self, f, node_id, insertion_order, node):
        self._items.append(
            {
                "f": f,
                "node_id": node_id,
                "insertion_order": insertion_order,
                "node": node,
            }
        )

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty PriorityQueue")

        best_index = 0
        best_item = self._items[0]
        best_key = (
            best_item["f"],
            best_item["node_id"],
            best_item["insertion_order"],
        )

        for index in range(1, len(self._items)):
            item = self._items[index]
            key = (item["f"], item["node_id"], item["insertion_order"])
            if key < best_key:
                best_index = index
                best_item = item
                best_key = key

        self._items.pop(best_index)
        return best_item["node"]

    def is_empty(self):
        return len(self._items) == 0

    def __len__(self):
        return len(self._items)