import cv2
import os


def makevideo(videoname, frames=5):

    image_folder = 'figs'
    video_name = videoname + '.mp4'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'3IVD')
    video = cv2.VideoWriter(video_name, fourcc, frames, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def delfigs():
    image_folder = 'figs'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    for image in images:
        os.remove(os.path.join(image_folder, image))