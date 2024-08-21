import matplotlib.pyplot as plt


class Simulator(object):
    '''Convenience class for simulating models and viewing the results.
    '''

    def __init__(self):
        pass


    def iterate(self, model, input_img, steps=5):
        '''
        '''
        self.img = input_img
        self.results = []

        for _ in range(steps):
            res = model.iterate(input_img)
            self.results.append(res)

          

    def plot_results_one_plot(self):
        '''
        '''
        nr_rows = 2
        nr_cols = 12
        nr_left = nr_rows * nr_cols - 2*2 # 2*2 from row_span + col_span
        nr_per_row = nr_left // nr_rows

        plt.subplot2grid((nr_rows, nr_cols), (0,0), rowspan=2, colspan=2)
        plt.imshow(self.img, cmap='gray')
        plt.xticks([])
        plt.yticks([])

        for n,i in enumerate(self.results):
            plt.subplot2grid((2,12), (n//nr_per_row, n%nr_per_row + 2))
            plt.imshow(i)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

        plt.show()


    def plot_results(self, step_size=15):
        plt.imshow(self.img, cmap='gray', extent=[0,100,0,1], aspect='auto')
        plt.colorbar()

        plt.show()

        for n,i in enumerate(self.results[::step_size]):
            plt.title(f"Iteration: {(n+1) * step_size}")
            plt.imshow(i, cmap='gray', extent=[0,100,0,1], aspect='auto')
            plt.colorbar()
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            

            plt.show()

        
