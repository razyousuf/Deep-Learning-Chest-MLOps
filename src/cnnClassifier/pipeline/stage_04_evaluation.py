from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier import logger


STAGE_NAME = "Evaluation"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        # Initialize the logger
        logger.info(f"=========================")
        logger.info(f">>>>>>>>>>>>>> Stage {STAGE_NAME} started. <<<<<<<<<<<<\n")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>>>> Stage {STAGE_NAME} completed. <<<<<<<<<<<<\n")
        logger.info(f"=========================")
    except Exception as e:
        logger.exception(e)
        raise e