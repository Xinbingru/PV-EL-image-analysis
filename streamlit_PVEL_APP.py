import os
import array
import time
from io import BytesIO
import sys
import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import torch
from IPython.display import display
from pathlib import Path
from tqdm import tqdm
from pv_vision.transform_crop.solarmodule import MaskModule
from ipywidgets import interactive, widgets, Layout
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from scipy.signal import find_peaks, find_peaks_cwt, argrelextrema
from scipy.ndimage import gaussian_filter1d


def main():

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Select your image type")
    app_mode = st.sidebar.selectbox("image type",
        ["Lab (PV cell)", "Field (PV module)"])
    if app_mode == "Lab (PV cell)":
        run_lab_app()
    elif app_mode == "Field (PV module)":
        run_field_app()

def run_lab_app():

    uploadfile = st.file_uploader(label="Upload image", type=['jpg', 'png', 'jpeg'])
    save_dir = './image/Lab_image'
    os.makedirs(save_dir, exist_ok = True)

    if uploadfile is not None:

        image = load_local_image(uploadfile)
        st.image(image)
        
        input_box, masks = box_input_SAM(image)
        SAM_mask_show(image, masks, input_box)
        masked_tr = SAM_image_save(image)

        SAM_image = image = np.array(Image.open(os.path.join(save_dir, uploadfile.name[0:-4] + '_ SAM_image.png')))
        image_transformed_1024 = transform_image(SAM_image)

        st.image(image_transformed_1024)

        if st.button('Rotate 90 degrees clockwise',help="ratate"):

            st.success('The rotation has been successful, please check whether the bus bar is vertical')
            img_rot_1024 = np.rot90(image_transformed_1024,-1)
            st.image(img_rot_1024)

        st.success('test check' )

    else: 
        st.write("Make sure you image is in JPG/PNG Format.")

def run_field_app():

    uploadfile = st.file_uploader(label="Upload image", type=['jpg', 'png', 'jpeg'])
    save_dir = './image/Field_image'
    os.makedirs(save_dir, exist_ok = True)

def process_time(num):

    word = st.empty()
    bar = st.progress(0)
    speed = round(100/num)
        
    for i in range(100):
        word.text('Image is processing : '+str(i+speed)+' %')
        bar.progress(str(i+speed))
        time.sleep(0.1)

def load_local_image(uploaded_file):

    # 原始图像 → Lab image/Field image
    # 过程图像 → process image
    # 结果图像 → result image

    bytes_data = uploaded_file.getvalue()  
    image = np.array(Image.open(BytesIO(bytes_data)))

    return image

def show_mask(mask, ax, random_color=False):
    # show SAM mask
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    # show SAM point-based input
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    # show SAM box-based input
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def box_input_SAM(image):
    # 图像预处理：读取，OTSU_二值化，canny边缘计算
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # _,binary_image = cv2.threshold(gray_img,40,256,cv2.THRESH_BINARY)
    _,binary_image = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = cv2.Canny(gray_img, threshold1=50, threshold2=150)

    # 计算max count，自动化计算input box
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found, please check the parameters for image and edge detection.")
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    box = [x - 10, y - 10 , x + w + 5, y + h + 5]
    input_box = np.array(box)

    # SAM workflow
    sam_checkpoint = "./checkpoint/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    predictor.set_image(gray_RGB_img)

    masks, scores, logits = predictor.predict(box=input_box)
    if masks is None or len(masks) == 0:
        raise ValueError("There is no mask, please check your box or model")
    
    return input_box, masks

def show_SAM_mask(image,masks,input_box):

    gray_RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    
    plt.imshow(gray_RGB_img,cmap='gray')
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off')

    st.pyplot(fig)
    
    # save_image_path = os.path.join(save_dir, uploadfile.name[0:-4] + '_ SAM_mask.png')
    # plt.savefig(save_image_path, transparent=True, dpi=300, pad_inches=0)
    # st.image(os.path.join(save_dir, uploadfile.name[0:-4] + '_ SAM_mask.png'))

def save_SAM_image(image):

    gray_RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = masks[0]
    mask = mask.astype(np.uint8) * 255
    masked_img = cv2.bitwise_and(gray_RGB_img , gray_RGB_img , mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_img = masked_img[y:y+h, x:x+w]

    tmp = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(cropped_img)
    rgba = [b,g,r, alpha]
    masked_tr = cv2.merge(rgba,4)
    save_image_path = os.path.join(save_dir_SAM, uploadfile.name[0:-4] + '_SAM_image.png')
    cv2.imwrite(save_image_path, masked_tr)

    # st.image(os.path.join(save_dir, uploadfile.name[0:-4] + '_SAM_image.png'),width=600)

    return masked_tr

def transform_image(SAM_image):

    image = cv2.cvtColor(SAM_image, cv2.COLOR_RGB2GRAY)
    raw_module = MaskModule(image, 1, 1, 8)
    def update_mask(thre):
        mask = raw_module.load_mask(thre=thre, output=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(mask, "gray")

    thre = widgets.FloatSlider(value=0.05, min=0, max=1, step=0.01)

    interactive(update_mask,
                thre=thre, description="Threshold")
    
    def update_corner(mode):
        corners = raw_module.corner_detection_cont(output=True, mode=mode)
        x, y = zip(*corners)

        plt.figure(figsize=(8, 6))
        plt.imshow(raw_module.mask, 'gray')
        plt.scatter(x, y, c='r',linewidths = 7)

    mode=widgets.IntSlider(value=4, min=0, max=4, step=1)
    interactive(update_corner,
                mode=mode)

    img_transformed_1024 = raw_module.transform(cellsize=1024, img_only=True)

    return img_transformed_1024

def thresholds(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,th1 = cv2.threshold(image,125,255,cv2.THRESH_BINARY)
    _,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,th3 = cv2.threshold(cv2.GaussianBlur(image,(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return th1, th2, th3

def find_image_peaks(threshold_image):

    mean_intensity = np.mean(threshold_image, axis=0)
    smoothed_intensity = mean_intensity
    smoothed_intensity = gaussian_filter1d(mean_intensity, sigma=2)
    inverse_intensity = - smoothed_intensity
    # peaks, _ = find_peaks(inverse_intensity, height= -50)
    # find peaks based on distance between on busbars
    # peaks, _ = find_peaks(inverse_intensity, distance = 115)

    # find peaks based on prominence
    peaks, _ = find_peaks(inverse_intensity, prominence=20)

    return smoothed_intensity, peaks

def show_peaks(smoothed_intensity, peaks):

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(smoothed_intensity, color='#333333', label='Smoothed Intensity')
    ax.plot(peaks, smoothed_intensity[peaks], "x", markersize='12', markeredgecolor='#66A61E' ,label='Dark Region Centers')
    ax.set_title('Pixel distribution based on the Y axis (green "X" indicate the location of the busbar)')
    ax.set_xlabel('Pixel Position along X-axis / (-)',fontsize=12)
    ax.set_ylabel('Average Pixel Intensity / (-)',fontsize=12)

    # show peaks
    st.pyplot(fig)

def find_busbars(image,peaks):

    mask_busbar = np.ones(image.shape[:2], dtype=np.uint8)

    lines_box_1 =[peaks[0]-2, 0, peaks[0]+2,1024]
    lines_box_1_np = np.array(lines_box_1)

    lines_boxes = []

    for i in range(len(peaks)):
        new_a = peaks[i]-2
        new_c = peaks[i]+2
        new_box = [new_a, 0, new_c, 1024]
        lines_boxes.append(new_box)
        mask_busbar[0:1024, new_a : new_c] = 0
        masked_image = cv2.bitwise_and(image, image, mask=mask_busbar)

    return lines_boxes, masked_image

def show_busbars(lines_boxes, masked_image):

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    plt.imshow(masked_image,'gray')

    for i, box_n in enumerate(lines_boxes):
        print(f'box_{i+1} = {box_n}')
        lines_box_n = np.array(box_n)
        show_box(lines_box_n, plt.gca())
        
    ax.set_title('Busbar image (green area is busbar)')
    plt.axis('off')

    st.pyplot(fig)

def calculate_corrosion_ratio(masked_image, peaks, busbar_range=0.25):

    # 设置左右范围（range）
    offset = math.ceil((peaks[1]-peaks[0])*busbar_range)


    analysis_img = masked_image

    background = 1
    retal = cv2.mean(analysis_img)
    # corrosion_thres = 125
    corrosion_thres = retal[0]

    number_of_total_pix = np.sum(analysis_img <= 255)
    number_of_busbar_pix = np.sum(analysis_img < background)
    number_of_good_pix = np.sum(analysis_img > corrosion_thres)
    # number_of_PV_pix
    number_of_PV_pix = np.sum(analysis_img <= 255) -  np.sum(analysis_img < 1)
    busbar_ratio = number_of_busbar_pix/number_of_total_pix*100
    corrosion_ratio = (1-number_of_good_pix/number_of_PV_pix)*100
    # print('total ratio of busbar is ', number_of_busbar_pix/number_of_total_pix, ', and total ratio of corrosion is ', 1-number_of_good_pix/number_of_total_pix)

    #create a mask based on thresholds
    corr_mask =np.zeros(np.shape(analysis_img))
    for i in range(0, np.shape(analysis_img)[0]):
        for j in range(0, np.shape(analysis_img)[1]):
            if (analysis_img[i,j] < background).all():
                corr_mask[i,j] = 255
            elif (analysis_img[i,j] > corrosion_thres).all():
                corr_mask[i,j] = 0
            else:
                corr_mask[i,j] = 125


    average_values = []
    corrosion_ratio_follow_busbar = []

    for peak in peaks:

        start_x = max(0, peak - offset)
        end_x = min(analysis_img.shape[1], peak + offset + 1)

        rectangle_region = analysis_img[:, start_x:end_x]

        mean_value = np.mean(rectangle_region)
        peaks_total_pix = np.sum(rectangle_region  <= 255)
        peaks_busbar_pix = np.sum(rectangle_region  < background)
        peaks_good_pix = np.sum(rectangle_region  > corrosion_thres)

        peaks_PV_pix = np.sum(rectangle_region <= 255) -  np.sum(rectangle_region < 1)
        peaks_busbar_ratio = peaks_busbar_pix/ peaks_total_pix*100
        peaks_corrosion_ratio = (1-peaks_good_pix/peaks_PV_pix)*100

        average_values.append(mean_value)
        corrosion_ratio_follow_busbar.append(peaks_corrosion_ratio)

    return corr_mask, busbar_ratio, corrosion_ratio, offset, average_values, corrosion_ratio_follow_busbar

def show_corrosion(masked_image, corr_mask, busbar_ratio, corrosion_ratio):

    st.markdown("""
                    Corrosion along the busbar location is calculated by setting a range threshold(:green[**offset=0.25**]) along the busbar.
                    
                    
                    """)

    # analysis_img_colored = cv2.cvtColor(np.asarray(masked_image), cv2.COLOR_BAYER_BG2BGR)
    corr_mask = corr_mask.astype(np.uint8)
    # 将掩膜转换为彩色图像
    colored_mask = cv2.applyColorMap(corr_mask, cv2.COLORMAP_JET)  # 使用伪彩色图像增强对比度
    # 调整透明度，alpha 表示原始图像的权重，beta 表示分段图像的权重
    alpha = 1  # 原始图像的透明度
    beta = 0.25  # 处理后图像的透明度
    blended_image = cv2.addWeighted(masked_image , alpha, colored_mask, beta, 0)
    # 显示图像
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.imshow(blended_image)
    ax.set_title('Corrosion image')

    for i, peak in enumerate(peaks):

        # plt.axvline(x=peak, color='green', linestyle='--')

        plt.text(peak, image.shape[0] // 7 , f'{corrosion_ratio_follow_busbar[i]:.2f} %',
                    color='black', fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='grey', alpha=0.8))

    fig.suptitle('Busbar: {:.2f}% | Corrosion: {:.2f}%'.format(busbar_ratio, corrosion_ratio))
    
    st.pyplot(fig)

def show_results_table(peaks, corrosion_ratio_follow_busbar,busbar_ratio,corrosion_ratio):

    busbars = peaks.tolist()
    corrosion_ratio_follow_busbar_2d = (np.round(corrosion_ratio_follow_busbar,2)).tolist()
    corrosion_ratio_follow_busbar_show = [str(i)+' %' for i in corrosion_ratio_follow_busbar_2d]

    data = {
            'Image Name': [uploadfile.name[0:-4]] ,
            'Image size': ['1024*1024'],
            'Busbar ratio (%)' : [f'{busbar_ratio:.2f} %'],
            'Overall corrosion ratio (%)' : [f'{corrosion_ratio:.2f} %'],
            'Busbar position': [busbars],
            'Corrosion ratio around the busbars' : [corrosion_ratio_follow_busbar_2d]
            }
    df = pd.DataFrame(data)

    data_show = {
            'Image':['Image Name', 'Image size','Busbar ratio','Overall corrosion ratio','Busbar position','Corrosion ratio around the busbars'],
            'Value':[uploadfile.name[0:-4], '1024*1024',f'{busbar_ratio:.2f} %', f'{corrosion_ratio:.2f} %',busbars, corrosion_ratio_follow_busbar_show]
            }
    df_show = pd.DataFrame(data_show)

    st.title("Image analysis calculations")
    st.write("Image: ", uploadfile.name)
    st.table(df_show)


# logo

Dow_logo_2 = "image/Dow.jpg"
Dow_logo = "image/dow-logo.png"
EPS_logo = "image/EPS-logo.png"
PV_EL_logo = "image/PV-EL_logo.png"

st.sidebar.image(Dow_logo, width=100)

col1, col2, col3 = st.columns(3)

with col1:
    st.image(Dow_logo, width=200)

with col2:
    st.image(EPS_logo, width=100)

# APP title
st.title("Welcome to PV-EL APP :red[V1.0]👋")
st.markdown("")

col3, col4= st.columns(2)

with col3:

    st.markdown(

        """
        The PV-EL image analysis APP is an application that integrates photovoltaic electroluminescence(PV-EL) image segmentation, 
        transformation, busbar recognition, and corrosion analysis.

        If you have any questions, feel free to contact us: 

        :green[**Xin, Bingru**] bxin1@dow.com ,
        :green[**Ma, Yan**] yma10@dow.com ,
        :green[**Liu, Yang**] liu.yang@dow.com

        """
        )
    
with col4:

    st.image(PV_EL_logo, width=350)

# 这里加入一些SAM的讲解，highlight

# st.markdown(
#         """
#
#        **👈 Select a image typy from the dropdown on the left** to process your images!
#
#        ### Want to learn more?
#
#        - Check out [streamlit.io](https://streamlit.io)
#        - Jump into our [documentation](https://docs.streamlit.io)
#        - Ask a question in our [community
#          forums](https://discuss.streamlit.io)
#
#        ### See more complex demos
#
#        - Use a neural net to [analyze the Udacity Self-driving Car Image
#          Dataset](https://github.com/streamlit/demo-self-driving)
#        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
#    """
#    )

st.sidebar.title("Select your image type")

app_mode = st.sidebar.selectbox("image type",
    ["Lab (PV cell)", "Field (PV module)"])

if app_mode == "Lab (PV cell)":

    uploadfile = st.file_uploader(label="Upload image", type=['jpg', 'png', 'jpeg'])
    save_dir_raw = './image/Lab_image/raw_image'
    save_dir_SAM = './image/Lab_image/SAM_image'
    save_dir_transform = './image/Lab_image/transform_image'
    os.makedirs(save_dir_raw, exist_ok = True)
    os.makedirs(save_dir_SAM, exist_ok = True)
    os.makedirs(save_dir_transform, exist_ok = True)

    if uploadfile is not None:

        image = load_local_image(uploadfile)

        st.image(image)

        with st.spinner("Please wait..."):
            input_box, masks = box_input_SAM(image)
            show_SAM_mask(image, masks, input_box)

        masked_tr = save_SAM_image(image)
        SAM_image = np.array(Image.open(os.path.join(save_dir_SAM, uploadfile.name[0:-4] + '_SAM_image.png')))
        img_transformed_1024 = transform_image(SAM_image)

        st.image(img_transformed_1024, width=400)

        st.markdown("""
                    
                    For subsequent buabar recognition and health analysis, 
                    make sure that the busbar of the image is along the Y-axis.

                    If the busbar is along the X-axis, please click the :red[**Rotate image**] button, 

                    and if the busbar is along the Y-axis, please click the :red[**Keep original state**] button.
                    
                    """)

        col_rotate, col_keep = st.columns(2)

        with col_rotate:
            rotate_button = st.button('Rotate image',help="Rotate 90 degrees")

        with col_keep:
            keep_button = st.button('Keep original state')

        save_image_path = os.path.join(save_dir_transform, uploadfile.name[0:-4] + '_transform_image.png')

        if rotate_button:
            image_rot = np.rot90(img_transformed_1024,-1)
            # image = image.rotate(90,expand=True)
            st.image(image_rot, width=400)
            cv2.imwrite(save_image_path, image_rot)
            transform_image = Image.open(os.path.join(save_dir_transform, uploadfile.name[0:-4] + '_transform_image.png'))
            BGR_transform_image = cv2.cvtColor(np.asarray(transform_image), cv2.COLOR_RGB2BGR)
            th1, th2, th3 = thresholds(BGR_transform_image)
            smoothed_intensity, peaks = find_image_peaks(th2)
            num_busbars = len(peaks)
            lines_boxes, masked_image = find_busbars(BGR_transform_image,peaks)
            show_peaks(smoothed_intensity, peaks)
            show_busbars(lines_boxes, masked_image)
            corr_mask, busbar_ratio, corrosion_ratio, offset, average_values, corrosion_ratio_follow_busbar = calculate_corrosion_ratio(masked_image, peaks, busbar_range=0.25)
            with st.spinner("Please wait..."):
                show_corrosion(masked_image, corr_mask, busbar_ratio, corrosion_ratio)
                show_results_table(peaks, corrosion_ratio_follow_busbar,busbar_ratio,corrosion_ratio)


        elif keep_button:
            image_rot = img_transformed_1024
            st.image(image_rot, width=400)
            cv2.imwrite(save_image_path, image_rot)
            transform_image = Image.open(os.path.join(save_dir_transform, uploadfile.name[0:-4] + '_transform_image.png'))
            BGR_transform_image = cv2.cvtColor(np.asarray(transform_image), cv2.COLOR_RGB2BGR)
            th1, th2, th3 = thresholds(BGR_transform_image)
            smoothed_intensity, peaks = find_image_peaks(th2)
            num_busbars = len(peaks)
            lines_boxes, masked_image = find_busbars(BGR_transform_image,peaks)
            show_peaks(smoothed_intensity, peaks)
            show_busbars(lines_boxes, masked_image)
            corr_mask, busbar_ratio, corrosion_ratio, offset, average_values, corrosion_ratio_follow_busbar = calculate_corrosion_ratio(masked_image, peaks, busbar_range=0.25)
            with st.spinner("Please wait..."):
                show_corrosion(masked_image, corr_mask, busbar_ratio, corrosion_ratio)
                show_results_table(peaks, corrosion_ratio_follow_busbar,busbar_ratio,corrosion_ratio)

    else: 
        st.write("Make sure you image is in JPG/PNG Format.")

elif app_mode == "Field (PV module)":

    st.image("image/Dow Relief Banner.jpg")

