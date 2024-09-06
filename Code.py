import webbrowser
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.togglebutton import ToggleButton
import sympy as sp
import statistics as stats
import numpy as np
import matplotlib.pyplot as plt
from kivy.graphics.texture import Texture
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image

class GraphScreen(Screen):
    def __init__(self, **kwargs):
        super(GraphScreen, self).__init__(**kwargs)
        self.main_layout = BoxLayout(orientation='vertical')

        func_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        func_layout.add_widget(Label(text='Function (e.g., x**2 + 3*x + 2):'))
        self.function_input = TextInput(hint_text='Enter function', multiline=False)
        func_layout.add_widget(self.function_input)
        self.main_layout.add_widget(func_layout)

        plot_button = Button(text='Plot Graph', size_hint=(1, 0.1))
        plot_button.bind(on_press=self.plot_graph)
        self.main_layout.add_widget(plot_button)

        self.plot_container = BoxLayout(size_hint=(1, 0.8))
        self.main_layout.add_widget(self.plot_container)

        self.add_widget(self.main_layout)

    def plot_graph(self, instance):
        try:
            function_str = self.function_input.text
            x = np.linspace(-10, 10, 400)
            y = eval(function_str.replace('^', '**'))

            fig, ax = plt.subplots()
            ax.plot(x, y)
            fig.canvas.draw()

            width, height = fig.canvas.get_width_height()
            buf = fig.canvas.tostring_rgb()

            image = Image()
            texture = Texture.create(size=(width, height))
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()
            image.texture = texture

            self.plot_container.clear_widgets()
            self.plot_container.add_widget(image)

        except Exception as e:
            self.plot_container.clear_widgets()
            self.plot_container.add_widget(Label(text=f'Error: {str(e)}'))

class DeterminantScreen(Screen):
    def __init__(self, **kwargs):
        super(DeterminantScreen, self).__init__(**kwargs)
        self.main_layout = BoxLayout(orientation='vertical')
        
        # Input for matrix size
        size_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        size_layout.add_widget(Label(text='Matrix size (n):'))
        self.size_input = TextInput(hint_text='Enter size n', multiline=False)
        size_layout.add_widget(self.size_input)
        self.main_layout.add_widget(size_layout)

        # Button to set matrix size
        size_button = Button(text='Set Size', size_hint=(1, 0.1))
        size_button.bind(on_press=self.set_matrix_size)
        self.main_layout.add_widget(size_button)

        # Scrollable area for matrix input
        self.scroll_view = ScrollView(size_hint=(1, 0.4))
        self.matrix_layout = GridLayout(cols=1, size_hint_y=None)
        self.scroll_view.add_widget(self.matrix_layout)
        self.main_layout.add_widget(self.scroll_view)

        # Button to calculate determinant
        calc_button = Button(text='Calculate Determinant', size_hint=(1, 0.1))
        calc_button.bind(on_press=self.calculate_determinant)
        self.main_layout.add_widget(calc_button)

        # Label to display result
        self.result_label = Label(text='Determinant: ', size_hint=(1, 0.1))
        self.main_layout.add_widget(self.result_label)

        self.add_widget(self.main_layout)

    def set_matrix_size(self, instance):
        try:
            self.n = int(self.size_input.text)
            self.matrix_layout.clear_widgets()
            self.matrix_layout.cols = self.n
            self.matrix_layout.rows = self.n
            self.matrix_layout.size_hint_y = None
            self.matrix_layout.height = 30 * self.n
            self.matrix_elements = []

            for i in range(self.n):
                row_elements = []
                for j in range(self.n):
                    element_input = TextInput(hint_text=f'[{i+1},{j+1}]', multiline=False, size_hint_y=None, height=30)
                    self.matrix_layout.add_widget(element_input)
                    row_elements.append(element_input)
                self.matrix_elements.append(row_elements)
        except ValueError:
            self.result_label.text = 'Please enter a valid size.'

    def calculate_determinant(self, instance):
        try:
            matrix = []
            for row_elements in self.matrix_elements:
                row = [float(element.text) for element in row_elements]
                matrix.append(row)

            determinant = self.calculate_determinant_recursive(matrix)
            self.result_label.text = f'Determinant: {determinant}'
        except ValueError:
            self.result_label.text = 'Please enter valid numbers.'

    def calculate_determinant_recursive(self, matrix):
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

        determinant = 0
        for c in range(len(matrix)):
            sub_matrix = [row[:c] + row[c+1:] for row in matrix[1:]]
            sign = (-1) ** c
            sub_det = self.calculate_determinant_recursive(sub_matrix)
            determinant += sign * matrix[0][c] * sub_det
        return determinant

class DifferentialScreen(Screen):
    def __init__(self, **kwargs):
        super(DifferentialScreen, self).__init__(**kwargs)
        self.main_layout = BoxLayout(orientation='vertical')

        function_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        function_layout.add_widget(Label(text='Function (e.g., x**2 + 3*x + 2):'))
        self.function_input = TextInput(hint_text='Enter function', multiline=False)
        function_layout.add_widget(self.function_input)
        self.main_layout.add_widget(function_layout)

        variable_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        variable_layout.add_widget(Label(text='Variable (e.g., x):'))
        self.variable_input = TextInput(hint_text='Enter variable', multiline=False)
        variable_layout.add_widget(self.variable_input)
        self.main_layout.add_widget(variable_layout)

        diff_button = Button(text='Calculate Differential', size_hint=(1, 0.1))
        diff_button.bind(on_press=self.calculate_differential)
        self.main_layout.add_widget(diff_button)

        self.diff_result_label = Label(text='Differential: ', size_hint=(1, 0.1))
        self.main_layout.add_widget(self.diff_result_label)

        self.add_widget(self.main_layout)

    def calculate_differential(self, instance):
        try:
            function_str = self.function_input.text
            variable_str = self.variable_input.text

            variable = sp.symbols(variable_str)
            function = sp.sympify(function_str)

            differential = sp.diff(function, variable)
            self.diff_result_label.text = f'Differential: {sp.pretty(differential)}'
        except Exception as e:
            self.diff_result_label.text = f'Error: {str(e)}'

class IntegralScreen(Screen):
    def __init__(self, **kwargs):
        super(IntegralScreen, self).__init__(**kwargs)
        self.main_layout = BoxLayout(orientation='vertical')

        function_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        function_layout.add_widget(Label(text='Function (e.g., x**2 + 3*x + 2):'))
        self.function_input = TextInput(hint_text='Enter function', multiline=False)
        function_layout.add_widget(self.function_input)
        self.main_layout.add_widget(function_layout)

        variable_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        variable_layout.add_widget(Label(text='Variable (e.g., x):'))
        self.variable_input = TextInput(hint_text='Enter variable', multiline=False)
        variable_layout.add_widget(self.variable_input)
        self.main_layout.add_widget(variable_layout)

        int_button = Button(text='Calculate Integral', size_hint=(1, 0.1))
        int_button.bind(on_press=self.calculate_integral)
        self.main_layout.add_widget(int_button)

        self.int_result_label = Label(text='Integral: ', size_hint=(1, 0.1))
        self.main_layout.add_widget(self.int_result_label)

        self.add_widget(self.main_layout)

    def calculate_integral(self, instance):
        try:
            function_str = self.function_input.text
            variable_str = self.variable_input.text

            variable = sp.symbols(variable_str)
            function = sp.sympify(function_str)

            integral = sp.integrate(function, variable)
            self.int_result_label.text = f'Integral: {sp.latex(integral)}'
        except Exception as e:
            self.int_result_label.text = f'Error: {str(e)}'

class ZerosScreen(Screen):
    def __init__(self, **kwargs):
        super(ZerosScreen, self).__init__(**kwargs)
        self.main_layout = BoxLayout(orientation='vertical')

        poly_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        poly_layout.add_widget(Label(text='Polynomial (e.g., x**2 - 4):'))
        self.poly_input = TextInput(hint_text='Enter polynomial', multiline=False)
        poly_layout.add_widget(self.poly_input)
        self.main_layout.add_widget(poly_layout)

        zero_button = Button(text='Calculate Zeros', size_hint=(1, 0.1))
        zero_button.bind(on_press=self.calculate_zeros)
        self.main_layout.add_widget(zero_button)

        self.zero_result_label = Label(text='Zeros: ', size_hint=(1, 0.1))
        self.main_layout.add_widget(self.zero_result_label)

        self.add_widget(self.main_layout)

    def calculate_zeros(self, instance):
        try:
            poly_str = self.poly_input.text

            variable = sp.symbols('x')
            polynomial = sp.sympify(poly_str)

            zeros = sp.solve(polynomial, variable)
            self.zero_result_label.text = f'Zeros: {zeros}'
        except Exception as e:
            self.zero_result_label.text = f'Error: {str(e)}'

class StatisticsScreen(Screen):
    def __init__(self, **kwargs):
        super(StatisticsScreen, self).__init__(**kwargs)
        self.main_layout = BoxLayout(orientation='vertical')

        obs_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        obs_layout.add_widget(Label(text='Observations (comma separated):'))
        self.obs_input = TextInput(hint_text='e.g., 1,2,3', multiline=False)
        obs_layout.add_widget(self.obs_input)
        self.main_layout.add_widget(obs_layout)

        freq_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        freq_layout.add_widget(Label(text='Frequencies (comma separated):'))
        self.freq_input = TextInput(hint_text='e.g., 2,3,1', multiline=False)
        freq_layout.add_widget(self.freq_input)
        self.main_layout.add_widget(freq_layout)

        calc_layout = GridLayout(cols=3, size_hint=(1, 0.2))
        mean_button = Button(text='Mean')
        mean_button.bind(on_press=self.calculate_mean)
        calc_layout.add_widget(mean_button)

        median_button = Button(text='Median')
        median_button.bind(on_press=self.calculate_median)
        calc_layout.add_widget(median_button)

        mode_button = Button(text='Mode')
        mode_button.bind(on_press=self.calculate_mode)
        calc_layout.add_widget(mode_button)

        stddev_button = Button(text='Std Dev')
        stddev_button.bind(on_press=self.calculate_stddev)
        calc_layout.add_widget(stddev_button)

        variance_button = Button(text='Variance')
        variance_button.bind(on_press=self.calculate_variance)
        calc_layout.add_widget(variance_button)

        self.main_layout.add_widget(calc_layout)

        self.result_label = Label(text='Result: ', size_hint=(1, 0.1))
        self.main_layout.add_widget(self.result_label)

        self.add_widget(self.main_layout)

    def parse_input(self):
        observations = list(map(float, self.obs_input.text.split(',')))
        frequencies = list(map(int, self.freq_input.text.split(',')))
        return observations, frequencies

    def calculate_mean(self, instance):
        try:
            observations, frequencies = self.parse_input()
            mean = stats.mean([obs for obs, freq in zip(observations, frequencies) for _ in range(freq)])
            self.result_label.text = f'Mean: {mean}'
        except Exception as e:
            self.result_label.text = f'Error: {str(e)}'

    def calculate_median(self, instance):
        try:
            observations, frequencies = self.parse_input()
            data = [obs for obs, freq in zip(observations, frequencies) for _ in range(freq)]
            median = stats.median(data)
            self.result_label.text = f'Median: {median}'
        except Exception as e:
            self.result_label.text = f'Error: {str(e)}'

    def calculate_mode(self, instance):
        try:
            observations, frequencies = self.parse_input()
            data = [obs for obs, freq in zip(observations, frequencies) for _ in range(freq)]
            mode = stats.mode(data)
            self.result_label.text = f'Mode: {mode}'
        except Exception as e:
            self.result_label.text = f'Error: {str(e)}'

    def calculate_stddev(self, instance):
        try:
            observations, frequencies = self.parse_input()
            data = [obs for obs, freq in zip(observations, frequencies) for _ in range(freq)]
            stddev = stats.stdev(data)
            self.result_label.text = f'Std Dev: {stddev}'
        except Exception as e:
            self.result_label.text = f'Error: {str(e)}'

    def calculate_variance(self, instance):
        try:
            observations, frequencies = self.parse_input()
            data = [obs for obs, freq in zip(observations, frequencies) for _ in range(freq)]
            variance = stats.variance(data)
            self.result_label.text = f'Variance: {variance}'
        except Exception as e:
            self.result_label.text = f'Error: {str(e)}'

class EquationSolverScreen(Screen):
    def __init__(self, **kwargs):
        super(EquationSolverScreen, self).__init__(**kwargs)
        self.main_layout = BoxLayout(orientation='vertical')

        num_eq_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        num_eq_layout.add_widget(Label(text='Number of equations:'))
        self.num_eq_input = TextInput(hint_text='Enter number of equations', multiline=False)
        num_eq_layout.add_widget(self.num_eq_input)
        self.main_layout.add_widget(num_eq_layout)

        num_var_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        num_var_layout.add_widget(Label(text='Number of variables:'))
        self.num_var_input = TextInput(hint_text='Enter number of variables', multiline=False)
        num_var_layout.add_widget(self.num_var_input)
        self.main_layout.add_widget(num_var_layout)

        set_button = Button(text='Set', size_hint=(1, 0.1))
        set_button.bind(on_press=self.set_equation_size)
        self.main_layout.add_widget(set_button)

        self.scroll_view = ScrollView(size_hint=(1, 0.4))
        self.equations_layout = GridLayout(cols=1, size_hint_y=None)
        self.scroll_view.add_widget(self.equations_layout)
        self.main_layout.add_widget(self.scroll_view)

        var_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        var_layout.add_widget(Label(text='Variables (comma separated):'))
        self.var_input = TextInput(hint_text='e.g., x,y,z', multiline=False)
        var_layout.add_widget(self.var_input)
        self.main_layout.add_widget(var_layout)

        solve_button = Button(text='Solve Equations', size_hint=(1, 0.1))
        solve_button.bind(on_press=self.solve_equations)
        self.main_layout.add_widget(solve_button)

        self.result_label = Label(text='Solution: ', size_hint=(1, 0.1))
        self.main_layout.add_widget(self.result_label)

        self.add_widget(self.main_layout)

    def set_equation_size(self, instance):
        try:
            self.num_eq = int(self.num_eq_input.text)
            self.num_var = int(self.num_var_input.text)
            self.equations_layout.clear_widgets()
            self.equations_layout.cols = 1
            self.equations_layout.size_hint_y = None
            self.equations_layout.height = 30 * self.num_eq
            self.equation_elements = []

            for i in range(self.num_eq):
                equation_input = TextInput(hint_text=f'Equation {i+1}', multiline=False, size_hint_y=None, height=30)
                self.equations_layout.add_widget(equation_input)
                self.equation_elements.append(equation_input)
        except ValueError:
            self.result_label.text = 'Please enter valid numbers for equations and variables.'

    def solve_equations(self, instance):
        try:
            equations = []
            for eq_input in self.equation_elements:
                eq_str = eq_input.text
                equations.append(sp.sympify(eq_str))

            variables = self.var_input.text.split(',')
            variables = [sp.symbols(var.strip()) for var in variables]

            solution = sp.solve(equations, variables)
            self.result_label.text = f'Solution: {solution}'
        except Exception as e:
            self.result_label.text = f'Error: {str(e)}'

class DeterminantDifferentialCalculatorApp(App):
    def build(self):
        self.title = 'Matrix Determinant, Differential, Integral, Zeros, Statistics, Equation Solver, and Graph Plotter'
        self.main_layout = BoxLayout(orientation='vertical')

        self.screen_manager = ScreenManager()

        self.determinant_screen = DeterminantScreen(name='determinant')
        self.differential_screen = DifferentialScreen(name='differential')
        self.integral_screen = IntegralScreen(name='integral')
        self.zeros_screen = ZerosScreen(name='zeros')
        self.statistics_screen = StatisticsScreen(name='statistics')
        self.equation_solver_screen = EquationSolverScreen(name='equation_solver')
        self.graph_screen = GraphScreen(name='graph')

        self.screen_manager.add_widget(self.determinant_screen)
        self.screen_manager.add_widget(self.differential_screen)
        self.screen_manager.add_widget(self.integral_screen)
        self.screen_manager.add_widget(self.zeros_screen)
        self.screen_manager.add_widget(self.statistics_screen)
        self.screen_manager.add_widget(self.equation_solver_screen)
        self.screen_manager.add_widget(self.graph_screen)

        self.main_layout.add_widget(self.screen_manager)

        menu_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        det_button = ToggleButton(text='Determinant', group='screens', state='down')
        det_button.bind(on_press=lambda x: self.switch_screen('determinant'))
        diff_button = ToggleButton(text='Differential', group='screens')
        diff_button.bind(on_press=lambda x: self.switch_screen('differential'))
        int_button = ToggleButton(text='Integral', group='screens')
        int_button.bind(on_press=lambda x: self.switch_screen('integral'))
        zeros_button = ToggleButton(text='Zeros', group='screens')
        zeros_button.bind(on_press=lambda x: self.switch_screen('zeros'))
        stats_button = ToggleButton(text='Statistics', group='screens')
        stats_button.bind(on_press=lambda x: self.switch_screen('statistics'))
        eq_solver_button = ToggleButton(text='Equation Solver', group='screens')
        eq_solver_button.bind(on_press=lambda x: self.switch_screen('equation_solver'))
        graph_button = ToggleButton(text='Graph', group='screens')
        graph_button.bind(on_press=lambda x: self.switch_screen('graph'))

        menu_layout.add_widget(det_button)
        menu_layout.add_widget(diff_button)
        menu_layout.add_widget(int_button)
        menu_layout.add_widget(zeros_button)
        menu_layout.add_widget(stats_button)
        menu_layout.add_widget(eq_solver_button)
        menu_layout.add_widget(graph_button)

        self.main_layout.add_widget(menu_layout)

        return self.main_layout

    def switch_screen(self, screen_name):
        self.screen_manager.current = screen_name

if __name__ == '__main__':
    DeterminantDifferentialCalculatorApp().run()
