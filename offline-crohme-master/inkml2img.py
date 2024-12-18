import pickle as p
import cv2, json, scipy.misc, math
import numpy as np
from skimage.draw import line
from skimage.morphology import thin
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET
from io import StringIO

def get_label(inkml_file_abs_path):
    label = ""
    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"
    for child in root:
        if (child.tag == doc_namespace + 'annotation') and (child.attrib == {'type': 'truth'}):
            return child.text
    return label

def get_traces_data(inkml_file_abs_path):
    traces_data = []
    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"

    # Stores traces_all with their corresponding id
    traces_all = [{'id': trace_tag.get('id'),
                   'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
                               for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                            else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
                               for axis_coord in coord.split(' ')] \
                        for coord in (trace_tag.text).replace('\n', '').split(',')]} \
                  for trace_tag in root.findall(doc_namespace + 'trace')]

    # Sort traces_all list by id to make searching for references faster
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')
    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
            label = traceGroup.find(doc_namespace + 'annotation').text
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                traceDataRef = int(traceView.get('traceDataRef'))
                single_trace = traces_all[traceDataRef]['coords']
                traces_curr.append(single_trace)
            traces_data.append({'label': label, 'trace_group': traces_curr})
    else:
        [traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]

    return traces_data

def inkml2img(input_path, output_path):
    fout = open(output_path.split('.')[0] + '.txt', 'w+')
    fout.write(get_label(input_path))
    fout.close()

    traces = get_traces_data(input_path)
    
    plt.figure(figsize=(8, 8), dpi=100)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Hide axes and spines
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['bottom'].set_visible(False)
    plt.axes().spines['left'].set_visible(False)
    
    # Plot the traces
    for elem in traces:
        ls = elem['trace_group']
        for subls in ls:
            data = np.array(subls)
            x, y = zip(*data)
            plt.plot(x, y, linewidth=2, c='black')
    
    # Save the image without axes/border
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.gcf().clear()

    # Now, let's preprocess the image to resize and normalize
    img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

    # Thresholding to binarize the image
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find the bounding box of the image
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped_img = img[y:y+h, x:x+w]

    # Resize the image to the desired size (e.g., 256x256)
    resized_img = cv2.resize(cropped_img, (256, 256), interpolation=cv2.INTER_AREA)

    # Normalize image: scale pixel values to range [0, 1]
    normalized_img = resized_img / 255.0

    # Save the normalized image
    cv2.imwrite(output_path, (normalized_img * 255).astype(np.uint8))  # Convert back to 0-255 range for saving
    print(f"Processed and saved: {output_path}")
