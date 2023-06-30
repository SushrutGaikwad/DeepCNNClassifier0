# Deep Classifier Project

## Workflow

1. Update config.yaml.
2. Update secrets.yaml (Optional step).
3. Update params.yaml.
4. Update the entity in the 'entities' directory.
5. Update the configuration manager in the directory 'src/config'.
6. Update the component in the 'components' directory.
7. Update the pipeline.
8. Test run the pipeline stage.
9. Run tox to test the package.
10. Update dvc.yaml.
11. Run the command `dvc repro` for running all the stages in the pipeline.