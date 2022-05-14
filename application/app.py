"""App made with Flask and flask-sqlalchemy"""

from pathlib import Path
import os
import sys

# absolute path of the curent folder
current_path = os.path.dirname(__file__)
#:allow import from outside the current directory
sys.path.append(os.path.join(current_path, os.pardir))
print(os.path.join(current_path, os.pardir))

import time
from extract_text import run_extract
from model_run import model_run
from Database.tables import *
from flask import render_template, url_for, request, redirect
from sqlalchemy import and_, func, sql, update
from datetime import datetime
import json

from werkzeug.utils import secure_filename
from run_grobid import runGrobid
import magic
import threading



# Set the mime object, used to verify that uploaded file are really pdf
mime = magic.Magic(mime=True)

# Create the application and allow request to the database
app = create_app()
app.app_context().push()

# uploading folder
app.config['UPLOAD_FOLDER'] = os.path.join(current_path, 'static', 'uploaded_file', 'temp_pdf')

@app.route('/', methods=['POST', 'GET'])
@app.route('/<error>/', methods=['POST', 'GET'])
def showSimilarity(error=''):
    """This function is called at the root of the url.

    :param str error: The error message to display on the page, optional.

    :rtype: html
    """
    return render_template('home.html', error=error)


@app.route('/database-research/', methods=['POST', 'GET'])
def dataResearch():
    """
    This function show the form to make a request on the database *paper*.

    All field of the form are optional, if the user doesn't select any value they will be take the 'default' value.

    When the user click on the *research* button, the function will redirect to the *showPaper* function.

    :returns: The html research form or redirect to *showPaper*
    
    """
    if request.method == 'POST':
        id = request.form['Id']
        title = request.form['Title']
        author = request.form['Author']
        journal = request.form['Journal']
        dateMin = request.form['date_min']
        dateMax = request.form['date_max']
        site = request.form.get('site')

        if request.form.get('checkDate'):
            checkDate='on'
        else:
            checkDate='off'

        redir = ''
        for attribut in [id,title, author, journal, dateMin, dateMax, checkDate]:
            if attribut:
                redir += '/'+attribut
            else:
                redir += '/default'

        if site:
            redir += '/' + site
        else:
            redir += '/all'
        return redirect('/table'+redir)
    else:
        return render_template('database-form.html', today = datetime.today().strftime('%Y-%m-%d'))

@app.route('/table/<id>/<title>/<author>/<journal>/<dateMin>/<dateMax>/<checkDate>/<site>/', methods=['POST', 'GET'])
def showPaper(id, title, author, journal, dateMin, dateMax, site, checkDate):
    """
    This function make a request to the database *paper* and display the answer in a table.

    :param str id: id of a paper in the database
    :param str title:
    :param str author:
    :param str journal:
    :param str dateMin: by default 01/01/0001
    :param str dateMax: by default the date of the day
    :param str site: specify if the paper come from arXiv or CiteSeerX, both by default
    :param str checkDate: specify if paper should be filtered by **dateMin** and **dateMax**

    :rtype: html
    """
    columns = Paper.__table__.columns.keys()

    query = sql.true()
    if checkDate == 'on':
        if dateMin == 'default' and dateMax != 'default':
            query = query & (func.date(getattr(Paper, 'Date')) <= dateMax)
        elif dateMin != 'default' and dateMax == 'default':
            query = query & (func.date(getattr(Paper, 'Date')) >= dateMin)
        elif dateMin != 'default' and dateMax != 'default':
            query = query & and_(func.date(getattr(Paper, 'Date')) >= dateMin, func.date(getattr(Paper, 'Date')) <= dateMax)

    if site != 'all':
        query = query & (getattr(Paper, 'SourceSite') == site)

    for (field, value) in zip(['Id', 'Title', 'Author', 'Journal'], [id, title, author, journal]):
        if value != 'default':

            query = query & getattr(Paper, field).ilike(f"%{value}%")

    papers = Paper.query.filter(query)


    return render_template('database.html', papers=papers, columns=columns)

@app.route('/topics/<type>/<version>', methods=['POST', 'GET'])
def showTopic(type, version):
    """
    If method is GET, show the word topic distribution. Topic word are stored in the databases WordTopicBody, WordTopicAbstractV1 and WordTopicAbstractV2.

    If method is POST, store 'topicName' and 'topicId' in the appropriate table

    :param type: Body or Abstract
    :param version: V, V1 or V2

    :rtype: html
    """
    if request.method == 'POST':
        topicName = request.form['topicName']
        topicId = request.form['topicId']
        print(topicName)
        print(topicId)
        if type == "Abstract":
            if version == "V1":
                WordTopicAbstractV1.query.filter(WordTopicAbstractV1.Id == topicId).update({WordTopicAbstractV1.Name : topicName})
            else:
                WordTopicAbstractV2.query.filter(WordTopicAbstractV2.Id == topicId).update({WordTopicAbstractV2.Name : topicName})
        else:
            WordTopicBody.query.filter(WordTopicBody.Id == topicId).update({WordTopicBody.Name : topicName})
        db.session.commit()
        redirection = '/topics/' + type +'/' + version
        return redirect(redirection)
    else:
        if type == "Abstract":
            if version == "V1":
                topics = WordTopicAbstractV1.query.order_by(WordTopicAbstractV1.Id).all()
                columns = WordTopicAbstractV1.__table__.columns.keys()
                action = '/topics/Abstract/V1'
            else:
                topics = WordTopicAbstractV2.query.order_by(WordTopicAbstractV2.Id).all()
                columns = WordTopicAbstractV2.__table__.columns.keys()
                action = '/topics/Abstract/V2'
        else:
            topics = WordTopicBody.query.order_by(WordTopicBody.Id).all()
            columns = WordTopicBody.__table__.columns.keys()
            action = '/topics/Body/V'


        return render_template('topics.html', topics=topics, columns=columns, loads = json.loads, action = action)

@app.route('/body-abstract/', methods=['POST', 'GET'])
def showBodyAbstractDistribution():
    """
    Show a plot of papers abstracts and bodies in the topic space, projected in 2 dimensions.

    :rtype: html
    """

    return render_template('body-abstract.html')


@app.route('/similarity/result/', methods=['POST', 'GET'])
def process_similarity():
    """
    Display a pdf paper found in the uploaded_file folder.

    Extract the body and abstract from this paper and calculate the topic distribution.

    Find similar paper in the database based on the topic distribution and display them as suggestion.

    :rtype: html
    """

    # take the first pdf in temp_pdf
    # it will be used for displaying it in the application
    file = os.listdir(os.path.join(current_path, 'static', 'uploaded_file', 'temp_pdf'))[0]

    file = url_for('static', filename=os.path.join('uploaded_file', 'temp_pdf', file))

    runGrobid()
    dest = os.path.join(current_path, 'static', 'uploaded_file')
    run_extract(os.path.join(dest, 'grobid'), os.path.join(dest, 'abstracts'), os.path.join(dest, 'bodies'))

    similar = model_run()

    result = []
    for idx in similar[:5]:
        temp_paper = Paper.query.filter(Paper.Id == idx[0]).all()[0]
        result.append([temp_paper.Title, temp_paper.DownloadLink, temp_paper.Author, idx[1]])

    return render_template('similarity-result.html', filename=file+'#toolbar=0', similar=result)


@app.route('/uploader/', methods = ['GET', 'POST'])
def upload_file():
    """
    This function is called when the user click on the **submit** button on the homepage.
    It should check that their is a file and that it is a pdf.



    :return: redirect on the homepage with a warning message if no file are selected, else redirect to *process_similarity*
    """
    #sys.path.append('/home/stsapoui/Project/LDA-similarity-Project-/application')
    sys.path.append(current_path)

    # We first want to empty the folders if their are remaining papers

    #os.remove('/home/stsapoui/Project/LDA-similarity-Project-/application/static/uploaded_file/abstracts/10.1.1.1.1503')

    for folder in os.listdir(os.path.join(current_path, 'static','uploaded_file')):
        print('folder', folder)
        for file in os.listdir(os.path.join(current_path, 'static', 'uploaded_file', folder)):
            print('     file', file)
            os.remove(os.path.join(current_path, 'static', 'uploaded_file', folder, file))


    # We verify if the user provided a file
    if request.method == 'POST':
      f = request.files['file']
      if not f:
          return redirect('/Please enter file/')

      f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))

      if mime.from_file(os.path.join(current_path,'static', 'uploaded_file', 'temp_pdf', f.filename)) != 'application/pdf':
          os.remove(os.path.join(current_path,'static', 'uploaded_file', 'temp_file', f.filename))
          return redirect('/similarity/Only pdf document allowed, please retry')
      else:
        return redirect(url_for(('process_similarity')))
    else:
       return 'error type of request'

@app.route('/documentation/', methods=['POST', 'GET'])
def documentation():
    """
    It should redirect to the documentation  for the whole project.

    :rtype: html
    """

    return redirect('http://localhost:8000/view/sphinx%20documentation/build/html/index.html')

# run the app
if __name__ == "__main__":
    app.run(port=5555, debug=True)