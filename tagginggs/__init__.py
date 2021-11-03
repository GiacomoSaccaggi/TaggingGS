# -*- coding: utf-8 -*-
"""
FLASK
"""


def run_tagging_app():
    """
    
    :return: flask app
    """
    import os
    from flask import Flask, render_template, request, flash, send_file
    import sys
    constant_app = {'subtitle': 'Tool for tagging',
                    'menu_len': 3, 'menu_name': ['Home', 'Analyzes', 'Load data'],
                    'menu_link': ['/', '/Analyzes', '/Load_data'],
                    # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                    'menu_symbol': ['fa-home', 'fa-gears', 'fa-download'], 'powered_by': 'Giacomo Saccaggi',
                    'footer': 'The contents of this website, including (but not limited to) all written material,'
                    'images, photos, and code, are protected under international copyright and'
                    'trademark laws. You may not copy, reproduce, modify, republish, transmit or'
                    'distribute any material from this site without express written permission.'}

    dir_path = os.path.dirname(os.path.abspath(__file__))
    app = Flask(__name__, template_folder=f'{dir_path}', static_folder=f'{dir_path}/Static')
    app.config["SECRET_KEY"] = "GSTag"
    sys.path.append(dir_path)

    @app.route("/")
    def index():
        return render_template('index.html',
                               info_text='This program allows you to easily tag texts and images and it is also '
                                         'possible to create Machine Learning models (supervised and unsupervised)'
                                         ' in order to have a base from which to start to classify the different '
                                         'objects you want to tag. The subject of the next releases is to implement'
                                         ' an automatic tagging in which you start tagging and then it is done '
                                         'automatically asking for confirmation for those in doubt.',
                               title='EASY TAGGING', subtitle=constant_app['subtitle'],
                               menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                               menu_link=constant_app['menu_link'],
                               # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                               menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                               footer=constant_app['footer']
                               )

    @app.route("/Analyzes")
    def render_analyzes():
        from .TaggingGS_basefun import _all_analyzes, _remove_analysis
        _remove_analysis()
        num_analyzes, token_analysis, title_analysis, \
            tot_tagged, tot_to_tag, percentage_analysis, typetag = _all_analyzes()
        return render_template('analyzes.html', num_analyzes=num_analyzes, token_analysis=token_analysis,
                               title_analysis=title_analysis, tot_tagged=tot_tagged, typetag=typetag,
                               tot_to_tag=tot_to_tag, percentage_analysis=percentage_analysis,
                               title='EASY TAGGING - Analyzes', subtitle=constant_app['subtitle'],
                               menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                               menu_link=constant_app['menu_link'],
                               # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                               menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                               footer=constant_app['footer']
                               )

    @app.route("/Load_data")
    def render_loading():
        return render_template('load_page.html', title='EASY TAGGING - load data', subtitle=constant_app['subtitle'],
                               menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                               menu_link=constant_app['menu_link'],
                               # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                               menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                               footer=constant_app['footer']
                               )

    @app.route("/upload_files", methods=['POST'])
    def upload_files():
        print(request.form)
        import secrets
        import shutil
        from .TaggingGS_basefun import _load_data, _save_file_json, _save_file_pkl
        analysis_token = secrets.token_hex(16)
        metadata = {'analysis_token': analysis_token}
        tagging_type = request.form['tagging_type']
        metadata['analysis_title'] = request.form['analysis_title']
        metadata['categories'] = list(set(str(request.form['categ']).split(';')[1:]))
        sep = request.form['sep']
        try:
            os.mkdir('upload_tmp')
        except FileExistsError:
            pass
        files = []
        for file in request.files.getlist('caricamenti'):
            files.append(f"upload_tmp/{file.filename}")
            file.save(f"upload_tmp/{file.filename}")
        print(files)
        to_tag, metadata = _load_data(files, tagging_type, metadata=metadata,
                                      sep=sep if tagging_type != 'img' else '',
                                      base64=True if tagging_type == 'img' else False)
        shutil.rmtree('upload_tmp')
        if type(to_tag) == str:
            flash(to_tag, "not success")
            return render_template('load_page.html', title='EASY TAGGING - load data',
                                   subtitle=constant_app['subtitle'],
                                   menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                                   menu_link=constant_app['menu_link'],
                                   # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                                   menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                                   footer=constant_app['footer']
                                   )
        else:
            _save_file_json(metadata, 'metadata', analysis_token)
            _save_file_pkl(to_tag, 'to_tag', analysis_token)
            _save_file_pkl({}, 'tagged', analysis_token)
            flash("Files uploaded and sent for tagging", "success")
            return render_template('index.html',
                                   info_text='Files uploaded and sent for tagging go to "Analyzes" to see it',
                                   title='EASY TAGGING', subtitle=constant_app['subtitle'],
                                   menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                                   menu_link=constant_app['menu_link'],
                                   # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                                   menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                                   footer=constant_app['footer']
                                   )

    @app.route('/tag_and_next', methods=['POST'])
    def req_tag_and_next():
        print(request.form)
        from .TaggingGS_basefun import _load_tag_files, _extract_to_tag, _save_file_pkl, _add_tagged
        token = request.form["token"]
        to_tag, tagged, tagging_type, categorie = _load_tag_files(token)
        tagged[str(request.form["fname"])] = request.form["category"]
        _add_tagged(token)
        _save_file_pkl(tagged, 'tagged', token)
        if tagging_type == 'img':
            print(tagging_type)
            # remove img if we do not use base64
        tag = _extract_to_tag(to_tag, tagged)
        if tag == 'All tagged':
            flash("All tagged", "success")
            return render_template('index.html', info_text='All tagged!',
                                   title='EASY TAGGING', subtitle=constant_app['subtitle'],
                                   menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                                   menu_link=constant_app['menu_link'],
                                   # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                                   menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                                   footer=constant_app['footer']
                                   )
        else:
            if tagging_type == 'img':
                img_display = 'block'
                # caricare le immagini in flask
                text = 'You load this image all copyright problem...'
                # if we use base64
                img_filepath = to_tag[tag]
            else:
                img_display = 'none'
                text = to_tag[tag]
                img_filepath = ''
            return render_template('tag_page.html', categorie=categorie, img_display=img_display, token=token,
                                   text=text, img_filepath=img_filepath, file_name=tag,
                                   title='EASY TAGGING - tagging data', subtitle=constant_app['subtitle'],
                                   menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                                   menu_link=constant_app['menu_link'],
                                   # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                                   menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                                   footer=constant_app['footer']
                                   )

    @app.route('/tag_and_explore', methods=['POST'])
    def req_tag_and_explore():
        print(request.form)
        from .TaggingGS_basefun import _load_tag_files, _save_file_pkl
        token = request.form["token"]
        to_tag, tagged, tagging_type, categorie = _load_tag_files(token)
        tagged[str(request.form["fname"])] = request.form["category"]
        _save_file_pkl(tagged, 'tagged', token)
        if tagging_type == 'img':
            print(tagging_type)
            # remove img if we do not use base64
        img_display = 'block' if tagging_type == 'img' else 'none'
        text = '' if tagging_type == 'img' else []
        file_name = []
        img_filepath = [] if tagging_type == 'img' else ''
        categorized = []
        num_analysis = 0
        for k, v in tagged.items():
            num_analysis += 1
            file_name.append(k)
            categorized.append(v)
            if tagging_type == 'img':
                # if we use base64
                img_filepath.append(to_tag[k])
            else:
                text.append(to_tag[k])
        return render_template('explore.html', categorie=categorie, img_display=img_display, categorized=categorized,
                               token=token, num_analysis=num_analysis,
                               text=text, img_filepath=img_filepath, file_name=file_name,
                               title='EASY TAGGING - Explore data', subtitle=constant_app['subtitle'],
                               menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                               menu_link=constant_app['menu_link'],
                               # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                               menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                               footer=constant_app['footer']
                               )

    @app.route('/explore', methods=['GET'])
    def explore_analysis():
        from .TaggingGS_basefun import _load_tag_files
        token = request.args["token"]
        to_tag, tagged, tagging_type, categorie = _load_tag_files(token)
        img_display = 'block' if tagging_type == 'img' else 'none'
        text = '' if tagging_type == 'img' else []
        file_name = []
        img_filepath = [] if tagging_type == 'img' else ''
        categorized = []
        num_analysis = 0
        for k, v in tagged.items():
            num_analysis += 1
            file_name.append(k)
            categorized.append(v)
            if tagging_type == 'img':
                # if we use base64
                img_filepath.append(to_tag[k])
            else:
                text.append(to_tag[k])
        return render_template('explore.html', categorie=categorie, img_display=img_display, categorized=categorized,
                               token=token, num_analysis=num_analysis,
                               text=text, img_filepath=img_filepath, file_name=file_name,
                               title='EASY TAGGING - Explore data', subtitle=constant_app['subtitle'],
                               menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                               menu_link=constant_app['menu_link'],
                               # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                               menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                               footer=constant_app['footer']
                               )

    @app.route('/tagging', methods=['GET'])
    def tagging_new():
        from .TaggingGS_basefun import _load_tag_files, _extract_to_tag
        token = request.args["token"]
        to_tag, tagged, tagging_type, categorie = _load_tag_files(token)
        tag = _extract_to_tag(to_tag, tagged)
        if tag == 'All tagged':
            flash("All tagged", "success")
            return render_template('index.html', info_text='All tagged!',
                                   title='EASY TAGGING', subtitle=constant_app['subtitle'],
                                   menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                                   menu_link=constant_app['menu_link'],
                                   # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                                   menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                                   footer=constant_app['footer']
                                   )
        else:
            if tagging_type == 'img':
                img_display = 'block'
                # caricare le immagini in flask
                text = 'You load this image all copyright problem...'
                # if we use base64
                img_filepath = to_tag[tag]
            else:
                img_display = 'none'
                text = to_tag[tag]
                img_filepath = ''
            return render_template('tag_page.html', categorie=categorie, img_display=img_display, token=token,
                                   text=text, img_filepath=img_filepath, file_name=tag,
                                   title='EASY TAGGING - tagging data', subtitle=constant_app['subtitle'],
                                   menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                                   menu_link=constant_app['menu_link'],
                                   # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                                   menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                                   footer=constant_app['footer']
                                   )

    @app.route('/download', methods=['GET'])
    def download_by_token():
        from .TaggingGS_basefun import _export_analysis
        file_download = _export_analysis(request.args["token"])
        # return redirect(url_for('render_analyzes')) make it in javascript
        return send_file(file_download)

    @app.route('/delete', methods=['GET'])
    def delete_by_token():
        from .TaggingGS_basefun import _all_analyzes
        num_analyzes, token_analysis, title_analysis,\
            tot_tagged, tot_to_tag, percentage_analysis, typetag = _all_analyzes(delete=request.args["token"])
        return render_template('analyzes.html', num_analyzes=num_analyzes, token_analysis=token_analysis,
                               title_analysis=title_analysis, tot_tagged=tot_tagged, typetag=typetag,
                               tot_to_tag=tot_to_tag, percentage_analysis=percentage_analysis,
                               title='EASY TAGGING - Analyzes', subtitle=constant_app['subtitle'],
                               menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                               menu_link=constant_app['menu_link'],
                               # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                               menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                               footer=constant_app['footer']
                               )

    @app.route('/unsupervised', methods=['GET'])
    def unsupervised_by_token():
        from .TaggingGS_basefun import _load_tag_files, _exist_in_analysis, _all_analyzes
        print(request.args)
        tipo = 'unsupervised'
        comment = ''
        token = request.args["token"]
        to_tag, tagged, tagging_type, categorie = _load_tag_files(token)
        if 'retrain' in request.args and request.args["retrain"] == 'y':
            _all_analyzes(deletemodel=token, tipo=tipo, stop='y')
        if not _exist_in_analysis(token, tipo):
            if tagging_type == 'img':
                from unsupervised_img import clustering_images
                if "classes" in request.args.keys():
                    clustering_images(to_tag=to_tag, analysis_token=token, classes=request.args["classes"])
                else:
                    clustering_images(to_tag=to_tag, analysis_token=token)
                comment = 'The model used to divide the clusters is a K-means starting from a pre-trained VGG16. ' \
                          'Then perform a Fine Tuning through Transfer Learning.'
            else:
                from unsupervised_text import LDAText
                lan = request.args["lan"].lower()
                model = LDAText(lan=lan, analysis_token=token)
                if "classes" in request.args.keys():
                    model.train_lda(to_tag=to_tag, classes=request.args["classes"])
                else:
                    model.train_lda(to_tag=to_tag)
                comment = 'The model used is a LatentDirichletAllocation with TFIDF Vectorizer.'

        return render_template('models.html', tipo=tipo, token_analysis=token, typetag=tagging_type,
                               title='EASY TAGGING - Analyzes', subtitle=constant_app['subtitle'], comment=comment,
                               menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                               menu_link=constant_app['menu_link'],
                               # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                               menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                               footer=constant_app['footer']
                               )

    @app.route('/supervised', methods=['GET'])
    def supervised_by_token():
        from .TaggingGS_basefun import _load_tag_files, _exist_in_analysis, _all_analyzes
        print(request.args)
        tipo = 'supervised'
        comment = ''
        token = request.args["token"]
        to_tag, tagged, tagging_type, categorie = _load_tag_files(token)
        if 'retrain' in request.args and request.args["retrain"] == 'y':
            _all_analyzes(deletemodel=token, tipo=tipo, stop='y')
        if not _exist_in_analysis(token, tipo):
            if tagging_type == 'img':
                from supervised_img import easy_net
                easy_net(to_tag=to_tag, tagged=tagged, analysis_token=token)
                comment = 'The model used a pre-trained VGG16 with a Fine Tuning performed through Transfer Learning.'
            else:
                from supervised_text import SpacyEmbeddingModel
                lan = request.args["lan"].lower()
                model = SpacyEmbeddingModel(lan=lan, analysis_token=token)
                model.training(to_tag=to_tag, tagged=tagged, categorie=categorie)
                comment = 'The model used a pre-trained Spacy Embedding with a Fine Tuning ' \
                          'performed through Transfer Learning.'

        return render_template('models.html', tipo=tipo, token_analysis=token, typetag=tagging_type,
                               title='EASY TAGGING - Analyzes', subtitle=constant_app['subtitle'], comment=comment,
                               menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                               menu_link=constant_app['menu_link'],
                               # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                               menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                               footer=constant_app['footer']
                               )

    @app.route('/downloadmodel', methods=['GET'])
    def downloadmodel_by_token():
        from .TaggingGS_basefun import _export_analysis
        file_download = _export_analysis(analysis_token=request.args["token"], tipo=request.args["tipo"])
        # return redirect(url_for('render_analyzes')) make it in javascript
        return send_file(file_download)

    @app.route('/deletemodel', methods=['GET'])
    def deletemodel_by_token():
        from .TaggingGS_basefun import _all_analyzes
        num_analyzes, token_analysis, title_analysis,\
            tot_tagged, tot_to_tag, percentage_analysis, typetag = _all_analyzes(deletemodel=request.args["token"],
                                                                                 tipo=request.args["tipo"])
        return render_template('analyzes.html', num_analyzes=num_analyzes, token_analysis=token_analysis,
                               title_analysis=title_analysis, tot_tagged=tot_tagged, typetag=typetag,
                               tot_to_tag=tot_to_tag, percentage_analysis=percentage_analysis,
                               title='EASY TAGGING - Analyzes', subtitle=constant_app['subtitle'],
                               menu_len=constant_app['menu_len'], menu_name=constant_app['menu_name'],
                               menu_link=constant_app['menu_link'],
                               # https://www.w3schools.com/icons/fontawesome_icons_webapp.asp
                               menu_symbol=['fa-home', 'fa-gears', 'fa-download'], powered_by='Giacomo Saccaggi',
                               footer=constant_app['footer']
                               )

    # app.run(port=port)
    return app
