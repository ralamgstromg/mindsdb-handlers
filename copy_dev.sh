cp -r *_handler /home/rcastro/Documentos/temporal/mindsdb/mindsdb/integrations/handlers
cp ml_exec_base.py /home/rcastro/Documentos/temporal/mindsdb/mindsdb/integrations/libs/ml_exec_base.py


# dentro del contenedor docker
cp -rf *_handler /mindsdb/mindsdb/integrations/handlers
 
pip install .[s3ngx_handler]