from interim_demo.features.canny_demo import extract_canny_edges_demo
from interim_demo.features.color_histogram_demo import extract_color_histogram_demo
from interim_demo.features.hog_demo import extract_hog_vector_demo
from interim_demo.features.sobel_edges_demo import extract_sobel_edges_demo
from interim_demo.launch_ui import launch_ui
from interim_demo.models.knn_hog_demo import run_knn_hog_one_vs_all, run_knn_one_vs_all
from interim_demo.models.lr_hog_one_vs_all_demo import run_lr_one_vs_all
from interim_demo.models.lr_organic_vs_inorganic_demo import run_lr_organic_vs_inorganic, run_lr_organic_vs_inorganic_persisted_model

menu_options = {
    1: ['HOG', extract_hog_vector_demo],
    2: ['Sobel Edges', extract_sobel_edges_demo],
    3: ['Canny Edges', extract_canny_edges_demo],
    4: ['Colour Histogram', extract_color_histogram_demo],
    5: ['KNN with pixels', run_knn_one_vs_all],
    6: ['KNN with HOG', run_knn_hog_one_vs_all],
    7: ['Logistic Regression optimal threshold persisted 5000 iteration model', run_lr_one_vs_all],
    8: ['Organic vs Inorganic LR with HOG not persisted', run_lr_organic_vs_inorganic],
    9: ['Organic vs Inorganic LR persisted 3000 iteration model', run_lr_organic_vs_inorganic_persisted_model],
    10: ['Launch UI', launch_ui],
    11: ['Exit', exit]
}


def print_menu_options():
    for key, value in menu_options.items():
        print("{0} - {1}".format(key, value[0]))


def run_menu_action(option):
    try:
        option = int(option)
        print("Running {0} demo...".format(menu_options.get(option)[0]))
        menu_options.get(option)[1]()
    except Exception as e:
        print("Invalid input or error!")
        print(e)


def main_demo():
    running = True
    print("\nInterim Demo")
    while (running):
        print_menu_options()

        choice = input("Select an option: ")
        run_menu_action(choice)


if __name__ == "__main__":
    main_demo()
