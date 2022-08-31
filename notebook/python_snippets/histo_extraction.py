# t0 = time.time()
# for img in images_with_classification:
#     channel_histograms = ColorHistogram(target_color_ranges=[TargetColorRange.R, TargetColorRange.G, TargetColorRange.B])
#     channel_histograms.process_image(img)
#     channel_histograms.save_to_directory(img, "data\\pre_processed\\color_histogram")

# t1 = time.time()

# color_histograms = ColorHistogram(target_color_ranges=[TargetColorRange.R, TargetColorRange.G, TargetColorRange.B])
#         color_histograms.process_image(img)

#         histoR = color_histograms.histograms[0]

#         x = [1 * i for i in range(histoR.size)]
#         axR = plt.subplot(3,2,2)
#         plt.text(0.02, 0.85, "R", fontweight="bold", transform=axR.transAxes)
#         plt.plot(x, histoR, color='red')
#         plt.fill_between(x, histoR, color='red')
#         plt.show()
