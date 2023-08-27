from experience_modeler.modeler.modeler import Modeler

if __name__ == '__main__':
    modeler = Modeler('conf/conf.yaml')
    modeler.bulk_copy_dataset()

    modeler.process_dataset()