from collections import deque

class buffer:
    def __init__(self, max_size):
      self.max_size = max_size
      self.queue = deque([])

    def __len__(self):
      return len(self.queue)

    def __iter__(self):
      return iter(self.queue)

    def append(self, obj):
      self.queue.append(obj)
      if len(self.queue) > self.max_size:
        return self.queue.pop()
      else:
        return None

    def clear(self):
      self.queue = deque([])