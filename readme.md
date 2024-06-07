[![DOI](https://zenodo.org/badge/809755915.svg)](https://zenodo.org/doi/10.5281/zenodo.11515535)

# Charts Generation for Research Paper

This repository contains code to generate charts for the associated research paper. The code is designed to produce visual representations of data used in the paper, aiding in the presentation and analysis of research findings.

## Usage:

### Building the Docker Image:
To build the Docker image, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the root directory of the cloned repository.
3. Open a terminal or command prompt.
4. Run the following command to build the Docker image:

    ```
    docker build -t my-deepc-app .
    ```

### Running the Docker Container:
Once the Docker image is built, you can run the Docker container using the following steps:

1. Make sure Docker is installed on your system.
2. Open a terminal or command prompt.
3. Run the following command to start the Docker container:

    ```
    docker run --rm my-deepc-app
    ```

This command will execute the code within the Docker container, generating the charts as specified.

## Contributing:
If you would like to contribute to this project, feel free to fork the repository and submit a pull request with your changes. We welcome any improvements or bug fixes!

## License:
This project is licensed under the [MIT License](LICENSE), which means you are free to use, modify, and distribute the code for both commercial and non-commercial purposes. See the LICENSE file for more details.
