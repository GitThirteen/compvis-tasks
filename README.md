# compvis-t12
Computer Vision coursework repo - University of Bath (Team 12)


Tasks to do:

Check acute seperation function behaviour for case 8

Function to view houghlines based on:
    # view all hough lines detected
    # lines = lines.squeeze()
    # for line in lines:
    #     rho, theta = line
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0 + 1000*(a))
    #     x2 = int(x0 - 1000*(-b))
    #     y2 = int(y0 - 1000*(a))

    #     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    # plt.imshow(img)
    # plt.show()

EXTRA: Implement auto canny / edge averaging to remove double edges (potentially making code more robust?) 
