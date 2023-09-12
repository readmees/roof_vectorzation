import numpy as np
import re
import functools
import torch as th
import cv2
from numba import jit
import matplotlib.pyplot as plt
import re
import matplotlib.pyplot as plt


def graph2line(junctions, adj_mtx, threshold=0.5):
    assert len(junctions) == len(adj_mtx)
    # assert np.allclose(adj_mtx, adj_mtx.transpose((0, 2, 1)), rtol=1e-2, atol=1e-2), f"{adj_mtx}"
    bs = len(junctions)
    lines = []
    scores = []
    for b in range(bs):
        junc = junctions[b]
        mtx = adj_mtx[b]
        num_junc = np.sum(junc.sum(axis=1) > 0)
        line = []
        score = []
        for i in range(num_junc):
            for j in range(i, num_junc):
                if mtx[i, j] > threshold:
                    line.append(np.hstack((junc[i], junc[j])))
                    score.append(mtx[i, j])
        scores.append(np.array(score))
        lines.append(np.array(line))

    return lines, scores

def transform_to_original_coordinates(x_resized, y_resized, width_scale, height_scale):
    x_original = x_resized * width_scale
    y_original = y_resized * height_scale
    return x_original, y_original

def draw_lines(imgs, lines, scores=None, width=2, save_lines=None, scales=(1, 1)):
    assert len(imgs) == len(lines)
    imgs = np.uint8(imgs)
    bs = len(imgs)
    if scores is not None:
        assert len(scores) == bs
    res = []
    for b in range(bs):
        img = imgs[b].transpose((1, 2, 0))
        line = lines[b]
        if scores is None:
            score = np.zeros(len(line))
        else:
            score = scores[b]
        img = img.copy()
        for (x1, y1, x2, y2), c in zip(line, score):
            if save_lines:
                # Resize x and y scales = (width_scale, height_scale)
                pt1 = transform_to_original_coordinates(x1, y1, scales[0], scales[1])
                pt2 = transform_to_original_coordinates(x2, y2, scales[0], scales[1])
                # Open (or create) the text file in append mode
                with open(f'{save_lines}_lines.txt', 'a') as f:
                    # Write the variable's value followed by a newline
                    f.write(f'point1:{pt1}, point2:{pt2}, score:{c}\n')
            pt1, pt2 = (round(x1), round(y1)), (round(x2), round(y2))
            c = tuple(cv2.applyColorMap(np.array(c * 255, dtype=np.uint8), cv2.COLORMAP_JET).flatten().tolist())
            img = cv2.line(img, pt1, pt2, c, width)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res.append(th.from_numpy(img.transpose((2, 0, 1))))

    return res


def draw_jucntions(hms, junctions, save_junction=None, scales=(1,1)):
    assert len(hms) == len(junctions)
    if hms.ndim == 3:
        imgs = np.uint8(hms * 255)
    else:
        imgs = np.uint8(hms)
    bs = len(imgs)
    res = []
    for b in range(bs):
        if hms.ndim == 3:
            img = cv2.cvtColor(imgs[b], cv2.COLOR_GRAY2BGR)
        else:
            img = np.array(imgs[b].transpose((1, 2, 0)))
        junc = junctions[b]
        junc = junc[junc.sum(axis=1) > 0.1]
        if hms.ndim == 3:
            score = hms[b][np.int32(junc[:, 1]), np.int32(junc[:, 0])]
        else:
            score = [1.] * len(junc)
        img = img.copy()
        for (x, y), c in zip(junc, score):
     
            
            c = tuple(cv2.applyColorMap(np.array(c * 255, dtype=np.uint8), cv2.COLORMAP_JET).flatten().tolist())
            if save_junction:
                # Open (or create) the text file in append mode
                with open(f'{save_junction}_junctions.txt', 'a') as f:
                    # Write the variable's value followed by a newline
                    # Resize x and y scales = (width_scale, height_scale)
                    xt, yt = transform_to_original_coordinates(x, y, scales[0], scales[1])
                    f.write(f'x:{xt}, y:{yt}, score:{c}\n')
            cv2.circle(img, (round(x), round(y)), 5, c, thickness=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res.append(th.from_numpy(img.transpose((2, 0, 1))))

    return res

def plot_from_files(junction_file_path, line_file_path, img_path, output_path):
    # Extracting junctions
    junctions = []
    try:
        with open(junction_file_path, 'r') as f:
            for line in f:
                x = float(re.search(r'x:(.*?),', line).group(1))
                y = float(re.search(r'y:(.*?),', line).group(1))
                junctions.append((x, y))
    except FileNotFoundError:
        pass

    # Extracting lines
    lines = []
    try:    
        with open(line_file_path, 'r') as f:
            for line in f:
                point1 = re.search(r'point1:\((.*?), (.*?)\)', line)
                point2 = re.search(r'point2:\((.*?), (.*?)\)', line)
                x1, y1 = float(point1.group(1)), float(point1.group(2))
                x2, y2 = float(point2.group(1)), float(point2.group(2))
                lines.append(((x1, y1), (x2, y2)))
    except FileNotFoundError:
        pass
    
    # Load and display image
    img = cv2.imread(img_path)
    plt.imshow(img)

    # Plot the junctions and lines on top of the image
    if lines:
        for (x1, y1), (x2, y2) in lines:
            plt.plot([x1, x2], [y1, y2], 'b-')
    if junctions:
        for x, y in junctions:
            plt.scatter(x, y, c='red')

    if not junctions and not lines:
        print('No lines or junctions are found :(')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()  # Close the plot to free up memory

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='BGR'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
                        np.tile(colors[label],
                                (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end + 1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):
    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "%s"' % d)
    return ret
