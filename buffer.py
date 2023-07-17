import torch


class BoxBuffer:
        def __init__(self, length, item):
            self.max_len = length
            self.buffer = [item]
            self.rolling_avg = item
            self.used = 0

        def __contains__(self, box):
            return torch.linalg.norm(self.rolling_avg - box).item() < 100

        def append(self, box):
            if len(self.buffer) == self.max_len:
                self.buffer.pop(0)
            self.buffer.append(box)
            self.rolling_avg = torch.mean(torch.stack(self.buffer))

        def get(self):
            return self.rolling_avg


class RollingAverageSmoothing:
        def __init__(self, smoothness=0.8, period=60, num_predictions=4):
            self.buffers = {}
            self.pred_num = num_predictions
            self.window = period
            self.smoothing_ratio = smoothness

        def process(self, boxes):
            is_used = []
            for box in boxes:
                buffer_found = False
                for buffer in self.buffers:
                    if box in buffer:
                        buffer.append(box)
                        is_used.append(buffer)
                        buffer_found = True
                if not buffer_found:
                    buffer = BoxBuffer(self.window, box)
                    self.buffers[buffer] = []
                    is_used.append(buffer)
            for buffer in self.buffers:                                                                                                                                 
                if buffer in is_used:
                    self.buffers[buffer].append(1)
                else:
                    self.buffers[buffer].append(0)
            buffer_scores = {}          
            for buffer, usage in self.buffers.items():
                buffer_scores[buffer] = sum(usage[:self.window])*1.0/min(len(usage), self.window)
            sorted_buffers = {k: v for k, v in sorted(buffer_scores.items(), key=lambda item: item[1], reverse=True)}
            ret_boxes = []
            for i in range(self.pred_num):
                if sorted_buffers[list(sorted_buffers.keys())[i]] >= self.smoothing_ratio:
                    ret_boxes.append(list(sorted_buffers.keys())[i].get())
            return ret_boxes
