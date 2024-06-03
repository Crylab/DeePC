import numpy as np
from scipy import sparse
import osqp


class DeepC:
    def __init__(self, params: dict):
        if "N" in params.keys():
            self.N = params["N"]
        else:
            raise Exception("Unknown size of the dataset")

        if "init_length" in params.keys():
            self.init_length = params["init_length"]
        else:
            raise Exception("Unknown length of initial conditions")

        if "finish_length" in params.keys():
            self.finish_length = params["finish_length"]
        else:
            raise Exception("Unknown length of prediction horizon")

        if "n_inputs" in params.keys():
            self.n_inputs = params["n_inputs"]
        else:
            raise Exception("Unknown number of inputs")

        if "n_outputs" in params.keys():
            self.n_outputs = params["n_outputs"]
        else:
            raise Exception("Unknown number of outpus")

        self.total_length = self.init_length + self.finish_length
        self.channels = self.n_inputs + self.n_outputs
        self.dataset_exists = False
        self.dataset_formulated = False
        self.init_cond_exists = False
        self.reference_exists = False
        self.criteria_exists = False

    def set_pred_horiz(self, new_horiz: int = 2):
        self.finish_length = new_horiz
        self.total_length = self.finish_length + self.init_length

    def set_init_horiz(self, new_horiz: int = 2):
        self.init_length = new_horiz
        self.total_length = self.finish_length + self.init_length

    def set_data(self, input, output):
        if len(input) != self.N:
            raise Exception("Inputs must content N experiments")

        if len(output) != self.N:
            raise Exception("Outpus must content N experiments")

        H = np.array([[[0.0] * self.total_length] * self.channels] * self.N)
        for i in range(0, self.N):
            one_input = input[i]
            if len(one_input) != self.n_inputs:
                raise Exception("Input must content n_inputs channels")
            for j in range(0, self.n_inputs):
                if len(one_input[j]) != self.total_length:
                    raise Exception(
                        "Each input channel must content total_length samples"
                    )
                H[i][j] = one_input[j]
            one_output = output[i]
            if len(one_output) != self.n_outputs:
                raise Exception("Output must content n_outputs channels")
            for j in range(0, self.n_outputs):
                if len(one_output[j]) != self.total_length:
                    raise Exception(
                        "Each output channel must content total_length samples"
                    )
                H[i][j + self.n_inputs] = one_output[j]
        self.dataset = H
        self.dataset_exists = True

    def set_init_cond(self, input, output):
        if len(input) != self.n_inputs:
            raise Exception("Init.cond. must content n_inputs channels")
        if len(output) != self.n_outputs:
            raise Exception("Init.cond. must content n_outpus channels")
        for each in input:
            if len(each) != self.init_length:
                raise Exception("Init.cond. must content init_length samples")
        for each in output:
            if len(each) != self.init_length:
                raise Exception("Init.cond. must content init_length samples")

        self.input_init = input
        self.output_init = output
        self.init_cond_exists = True

    def dataset_reformulation(self, dataset_in):
        if not self.dataset_exists:
            raise Exception("Attempt to reformulate H without dataset")
        self.H = [[0.0] * (self.total_length * self.channels)] * self.N
        for i in range(0, self.N):
            chunk = np.array([])
            for j in range(0, self.channels):
                chunk = np.hstack((chunk, dataset_in[i][j][0 : self.init_length]))
            for j in range(0, self.channels):
                chunk = np.hstack((chunk, dataset_in[i][j][self.init_length :]))
            self.H[i] = chunk
        self.H = np.array(self.H)
        self.dataset_formulated = True

    def set_reference(self, reference):
        if len(reference) != self.n_outputs:
            raise Exception("Reference must content n_outpus channels")
        for each in reference:
            if len(each) != self.finish_length:
                raise Exception("Reference must content finish_length samples")
        self.reference = reference
        self.reference_exists = True

    def set_opt_criteria(self, criteria_in: dict):
        criteria = criteria_in
        if "R" in criteria.keys():
            if len(criteria["R"]) != self.n_inputs:
                raise Exception("Criterias must be provided for n_inputs channels")
            for each in criteria["R"]:
                if each < 0.0:
                    raise Exception(
                        "Optimization criteria cannot be negative or equal to zero"
                    )
        else:
            raise Exception("Unknown optimization criterias for inputs")

        if "Q" in criteria.keys():
            if len(criteria["Q"]) != self.n_outputs:
                raise Exception("Criterias must be provided for n_outputs channels")
            for each in criteria["Q"]:
                if each < 0.0:
                    raise Exception(
                        "Optimization criteria cannot be negative or equal to zero"
                    )
        else:
            raise Exception("Unknown optimization criterias for outputs")

        if "lambda_y" in criteria.keys():
            if len(criteria["lambda_y"]) != self.n_outputs:
                raise Exception("Criterias must be provided for n_outputs channels")
            for each in criteria["lambda_y"]:
                if each < 0.0:
                    raise Exception(
                        "Optimization criteria cannot be negative or equal to zero"
                    )
        else:
            raise Exception("Unknown optimization criterias for lambda_y")

        if "lambda_g" not in criteria.keys():
            raise Exception("Unknown optimization criterias for lambda_g")
        else:
            if criteria["lambda_g"] < 0.0:
                raise Exception(
                    "Optimization criteria cannot be negative or equal to zero"
                )

        if "beta_R" in criteria.keys():
            if len(criteria["beta_R"]) != self.n_inputs:
                raise Exception("Beta criterias must be provided for n_inputs channels")
            for each in criteria["beta_R"]:
                if each < 0.0 or each > 1.0:
                    raise Exception("Beta optimization criteria must be within [0, 1]")
        else:
            criteria["beta_R"] = np.array([1.0] * self.n_inputs)

        if "beta_Q" in criteria.keys():
            if len(criteria["beta_Q"]) != self.n_outputs:
                raise Exception(
                    "Beta criterias must be provided for n_outputs channels"
                )
            for each in criteria["beta_Q"]:
                if each < 0.0 or each > 1.0:
                    raise Exception("Beta optimization criteria must be within [0, 1]")
        else:
            criteria["beta_Q"] = np.array([1.0] * self.n_outputs)

        if "beta_lambda_y" in criteria.keys():
            if len(criteria["beta_lambda_y"]) != self.n_outputs:
                raise Exception(
                    "Beta criterias must be provided for n_outputs channels"
                )
            for each in criteria["beta_lambda_y"]:
                if each < 0.0 or each > 1.0:
                    raise Exception("Beta optimization criteria must be within [0, 1]")
        else:
            criteria["beta_lambda_y"] = np.array([1.0] * self.n_outputs)

        self.criteria = criteria
        self.criteria_exists = True

    def solve_raw(self):
        if not self.dataset_formulated:
            raise Exception("Attempt to solve the problem without formulated dataset")
        if not self.init_cond_exists:
            raise Exception("Attempt to solve the problem without initial conditions")
        if not self.reference_exists:
            raise Exception("Attempt to solve the problem without reference")
        if not self.criteria_exists:
            raise Exception("Attempt to solve the problem without criterias")
        A0 = -1.0 * np.identity(self.channels * self.finish_length)
        B0 = np.zeros(
            (self.channels * self.init_length, self.channels * self.finish_length)
        )
        A1 = np.concatenate((B0, A0)).T

        A2 = np.zeros(
            (self.n_inputs * self.init_length, self.n_outputs * self.init_length)
        )
        B2 = -1.0 * np.identity(self.n_outputs * self.init_length)
        C2 = np.zeros(
            (self.channels * self.finish_length, self.n_outputs * self.init_length)
        )
        A3 = np.concatenate((A2, B2, C2)).T

        B1 = np.concatenate((self.H, A1, A3))

        C1 = np.concatenate(
            (
                np.concatenate(self.input_init),
                np.concatenate(self.output_init),
                np.zeros(self.channels * self.finish_length),
            )
        )

        com = (
            self.N
            + (self.channels * self.finish_length)
            + (self.n_outputs * self.init_length)
        )
        D0 = self.criteria["lambda_g"] * np.identity(com)
        for input_iter in range(0, self.n_inputs):
            counter = -1
            for iterator in range(
                self.N + (input_iter * self.finish_length),
                self.N + ((input_iter + 1) * self.finish_length),
            ):
                counter += 1
                D0[iterator][iterator] = self.criteria["R"][input_iter] * (
                    self.criteria["beta_R"][input_iter] ** counter
                )
        for output_iter in range(0, self.n_outputs):
            counter = -1
            for iterator in range(
                self.N + ((input_iter + output_iter + 1) * self.finish_length),
                self.N + ((input_iter + output_iter + 2) * self.finish_length),
            ):
                counter += 1
                D0[iterator][iterator] = self.criteria["Q"][output_iter] * (
                    self.criteria["beta_Q"][output_iter] ** counter
                )
        for output_iter in range(0, self.n_outputs):
            counter = -1
            pre = self.N + (self.channels * self.finish_length)
            for iterator in range(
                pre + (output_iter * self.init_length),
                pre + ((output_iter + 1) * self.init_length),
            ):
                counter += 1
                D0[iterator][iterator] = self.criteria["lambda_y"][output_iter] * (
                    self.criteria["beta_lambda_y"][output_iter] ** counter
                )

        E1 = np.copy(self.reference)
        for i in range(0, self.n_outputs):
            E1[i] *= -self.criteria["Q"][i]
        E2 = np.concatenate(E1)
        F1 = np.zeros(self.N + (self.n_inputs * self.finish_length))
        F2 = np.zeros(self.n_outputs * self.init_length)
        E0 = np.concatenate((F1, E2, F2))
        self.solver = osqp.OSQP()
        D0_sparse = sparse.csc_matrix(D0)
        B1_sparse = sparse.csc_matrix(B1.T)
        self.solver.setup(D0_sparse, E0, B1_sparse, C1, C1, verbose=False)
        results = self.solver.solve()
        if results.info.status_val != 1:
            raise Exception("Problem is unfeasible")
        return results

    def solve(self):
        results = self.solve_raw()
        solution = np.array([[0.0] * self.finish_length] * self.n_inputs)
        for input_iter in range(0, self.n_inputs):
            solution[input_iter] = results.x[
                self.N
                + (input_iter * self.finish_length) : self.N
                + ((input_iter + 1) * self.finish_length)
            ]
        return solution
