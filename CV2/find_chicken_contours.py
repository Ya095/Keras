import cv2


class FildChickenContours:
    def __init__(self, img_path, threshold_value=80, figure="rectangle"):
        self.__img_path = img_path
        self.__threshold_value = threshold_value
        self.__figure = figure
        self.__color = (255, 255, 255)


    def __return_real_image(self):
        img_real = cv2.imread(self.__img_path)
        return img_real

    def __preprocessing_image(self):
        img = self.__return_real_image()
        img_gauss_blur = cv2.GaussianBlur(img.copy(), (3, 3), 0)
        img_gray = cv2.cvtColor(img_gauss_blur, cv2.COLOR_BGR2GRAY)
        return img_gray

    def __find_contours(self):
        img_gray = self.__preprocessing_image()
        ret, threshold_img = cv2.threshold(img_gray, self.__threshold_value, 255, cv2.THRESH_BINARY)

        con, hir = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return con

    def __create_figure_on_image(self, x, y, w, h, img):
        if self.__figure == "rectangle":
            img = cv2.rectangle(img, (x, y), (x + w, y + h), self.__color, thickness=2)
            return img
        if self.__figure == "circle":
            img = cv2.circle(img, (x + w // 2, y + h // 2), h // 2, self.__color, thickness=2)
            return img

    def __draw_contours(self):
        img_real = self.__return_real_image()
        con = self.__find_contours()

        for c in con:
            x, y, w, h = cv2.boundingRect(c)
            img = self.__create_figure_on_image(x, y, w, h, img_real)

        return img

    def show_image(self):
        img_res = self.__draw_contours()

        cv2.imshow("", img_res)
        cv2.waitKey(0)


image = FildChickenContours("../images/imgpsh_fullsize_anim.jpg", 90, figure="circle")
image.show_image()
