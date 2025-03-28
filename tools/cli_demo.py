import tyro

from sam2_depthanything.api.process_data import ProcessConfig, process_data

if __name__ == "__main__":
    process_data(tyro.cli(ProcessConfig))
