from flask import Blueprint, render_template, request
from flask_login import login_required
from database import db
from base.models import data_table_1
from base.models import data_table_2
import datetime
import pandas as pd

blueprint = Blueprint(
    'tables_blueprint',
    __name__,
    url_prefix='/tables',
    template_folder='templates',
    static_folder='static'
)


def retrieve_data(data_obect):
    raw_data = [i for i in data_obect.query.all()]
    legend = 'Monthly Data'
    labels = []
    values = []

    for u in data_obect.query.all():
        print(u.name, u.value1)
        values.append(u.value1)
        labels.append(u.name)

    df =pd.DataFrame()
    df["labels"] = labels
    df["values"] = values
    df["raw_data"] = raw_data
    df["legend"] = legend

    return (df)


@blueprint.route('/<template>')
@login_required
def route_template(template):
    df = retrieve_data(data_table_1)
    return render_template(template + '.html'
                           ,data_from_table_1 = df["raw_data"]
                           ,data_from_table_2 = df["raw_data"]
                           ,values=df["values"], labels=df["labels"], legend=df["legend"]
                           )




#@blueprint.route('/<template>_handle_data', methods=['GET', 'POST'])
@blueprint.route('/<template>', methods=['GET', 'POST'])
@login_required
def route_template_send_data(template):
    # Request data from form
    data_table_1_id = request.form['data_table_1_id']

    # Build new insert string that isnerts data into database
    newdata = data_table_1(data_table_1_id,'jhfj2','gys', 100)
    db.session.add(newdata)
    db.session.commit()

    # retrieve new data
    df = retrieve_data(data_table_1)

    return render_template(template + '.html'
                           ,data_from_table_1 = df["raw_data"]
                           ,data_from_table_2 = df["raw_data"]
                           ,values=df["values"], labels=df["labels"], legend=df["legend"]
                           )




