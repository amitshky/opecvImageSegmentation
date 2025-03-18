import cv2 as cv


def main():
    img = cv.imread("assets/eq2_rotated.jpg")
    assert img is not None, "Failed to load image"

    # keeping a copy of the image just in case
    img_original = img.copy()

    grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(grayscaled, (55, 55), 0)
    inverted = cv.bitwise_not(blurred)
    _, binarized = cv.threshold(inverted, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, _ = cv.findContours(binarized, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours_img = cv.drawContours(img, contours, -1, (255, 0, 255), 3)
    img_rect = img_original.copy()
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        img_rect = cv.rectangle(img_rect, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # cv.imshow("original image", img_original)
    # cv.imshow("grayscale", grayscaled)
    # cv.imshow("blurred", blurred)
    # cv.imshow("invert", inverted)
    # cv.imshow("binarization", binarized)
    # cv.imshow("contours", contours_img)
    cv.imshow("image with rectangles", img_rect)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return


if __name__ == "__main__":
    main()