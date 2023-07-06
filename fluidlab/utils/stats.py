import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_mse(imgs, ref_img):
    diff = (imgs - np.expand_dims(ref_img,0)) ** 2
    loss = diff.sum(axis=(1,2,3))
    return loss

def calculate_phys_stats(visc_dir, ref_img_path):
    os.makedirs(f"./goals/{visc_dir.split('/')[-1]}", exist_ok=True)
    output_imgs = []
    inj_values = []
    base_values = []
    ref_img = plt.imread(ref_img_path) / 255
    inj_imgs = []
    base_imgs = [] 
    for root, dirs, files in os.walk(visc_dir):
    # Check if "output.png" exists in the current directory
        if "output.png" in files:
            # If the file exists, add its absolute path to the output_files list
            f_name = os.path.join(root, "output.png")
            img = plt.imread(os.path.join(root, "output.png")) / 255
            s = f_name.split("/")[-2].split("_")
            if float(s[2]) == 0:
                inj_imgs.append(img)
                inj_values.append(float(s[-1]))
            if float(s[-1]) == 0:
                base_imgs.append(img)
                base_values.append(float(s[2]))
            output_imgs.append(img)
    output_imgs = np.array(output_imgs)
    inj_imgs = np.array(inj_imgs)
    base_imgs = np.array(base_imgs)
    if len(inj_values) != 0:
        inj_losses = calculate_mse(inj_imgs, ref_img)
        inj_values = np.array(inj_values)
        plt.scatter(inj_values, inj_losses)
        plt.xlabel(r'$\mu_i$')
        plt.ylabel(r'squared error')
        plt.title('Sensitivity to Ice Cream Viscosity')
        plt.savefig(f"./goals/{visc_dir.split('/')[-1]}/ice_cream.png")
        plt.clf()
    if len(base_values) != 0:
        base_losses = calculate_mse(base_imgs, ref_img)
        base_values = np.array(base_values)
        plt.scatter(base_values, base_losses)
        plt.xlabel(r'$\mu_c$')
        plt.ylabel(r'squared error')
        plt.title('Sensitivity to Coffee Viscosity ($\mu_m = 0$)')
        plt.savefig(f"./goals/{visc_dir.split('/')[-1]}/coffee.png")
    
calculate_phys_stats("./tmp/recorder/visc_stir", "./goals/icecream_goal.png")