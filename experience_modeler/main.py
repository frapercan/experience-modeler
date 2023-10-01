from experience_modeler.modeler.modeler import Modeler

if __name__ == '__main__':
    modeler = Modeler('conf/conf.yaml')
    modeler.bulk_copy_dataset()
    modeler.preprocess_dataset()
    modeler.flatten_samples_multiple_dir()
    modeler.generate_tar_iterable()
    modeler.generate_metadata()
