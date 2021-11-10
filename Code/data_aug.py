def readImgs(img_path):
  """Read input images

  Args:
      img_path ([str]): [image path]

  Returns:
      [list]: [List with cv2 images]
  """
  path = sorted(os.listdir(img_path))
  imgs = []
  for i in range(len(img_path)):
    img = cv2.imread(path[i])
    imgs.append(img)
  return imgs


def img2tensor(arg):
  """Convert Cv2 images to tensors

  Args:
      arg ([cv2 image]): []

  Returns:
      [tf.tensor]: [image tensor]
  """
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  arg = arg/255
  return arg


def GImg(imgs):
  """Transform a list og images to a list of tensors

  Args:
      imgs ([list]): [list with images]

  Returns:
      [list]: [list with image tensors]
  """
  Tens = []
  for i in range(len(imgs)):
    im = imgs[i]
    tensor = img2tensor(im)
    Tens.append(tensor)
  return Tens


def randomBr(imgs):
    rb = []
    for i in range(len(imgs)):
        seed = (i, 0)
        srb = tf.image.stateless_random_brightness(
                imgs[i] ,max_delta=0.4, seed=seed)
        rb.append(srb)
    return rb


def randomSat(imgs):
    sat = []
    for i in range(len(imgs)):
        seed = (i,0)
        satu = tf.image.stateless_random_saturation(ten_imgs[i], 0.5, 1.0, seed)
        sat.append(satu)
    return sat


def rgb2gray(imgs):
    gscale = []
    for i in range(len(imgs)):
        gray = tf.image.rgb_to_grayscale(imgs[i])
        gscale.append(gray)
    return gscale


def crop(imgs):
    crp = []
    for i in range(len(imgs)):
        cropped = tf.image.central_crop(imgs[i], central_fraction=0.5)
        crp.append(cropped)
    return crp


def randomCont(imgs):
    con = []
    for i in range(len(imgs)):
        seed = (i,0)
        cont = tf.image.stateless_random_contrast(imgs[i],0.2,0.5,seed=seed)
        con.append(cont)
    return con