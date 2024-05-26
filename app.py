import os
from flask import Flask, render_template, request, redirect
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from matplotlib.patches import Ellipse
import time

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Initialize Dash app
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = CustomData(
            T1=int(request.form['T1']),
            T2=int(request.form['T2']),
            T3=int(request.form['T3']),
            T4=int(request.form['T4']),
            T5=int(request.form['T5']),
            T6=int(request.form['T6']),
            T7=int(request.form['T7']),
            T8=int(request.form['T8']),
            T9=int(request.form['T9']),
            T10=int(request.form['T10']),
            T11=int(request.form['T11']),
            T12=int(request.form['T12']),
            T13=int(request.form['T13']),
            T14=int(request.form['T14']),
            T15=int(request.form['T15']),
            T16=int(request.form['T16']),
            T17=int(request.form['T17']),
            T18=int(request.form['T18']),
        )

        df = data.get_dataframe()
        predict_pipeline = PredictPipeline()
        cluster_label, data_pca = predict_pipeline.predict(df)
        train_pca = np.load('artifacts/train_pca.npy')
        train_clusters = np.load('artifacts/train_clusters.npy')

        # Save data for Dash visualization
        np.save('artifacts/data_pca.npy', data_pca)
        np.save('artifacts/cluster_label.npy', cluster_label)

        return redirect('/dash/')

    return render_template('predict.html')

# Dash layout and callback
dash_app.layout = html.Div(style={'width': '100%', 'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}, children=[
    html.H1("Cluster Visualization", style={'textAlign': 'center'}),
    dcc.Graph(id='cluster-plot', style={'width': '80vw', 'height': '80vh'}),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

@dash_app.callback(
    Output('cluster-plot', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    train_pca = np.load('artifacts/train_pca.npy')
    train_clusters = np.load('artifacts/train_clusters.npy')
    data_pca = np.load('artifacts/data_pca.npy')
    cluster_label = np.load('artifacts/cluster_label.npy')

    fig = go.Figure()

    # Add scatter plot for training data with clusters
    unique_clusters = np.unique(train_clusters)
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    for i, cluster in enumerate(unique_clusters):
        cluster_points = train_pca[train_clusters == cluster]
        fig.add_trace(go.Scatter(
            x=cluster_points[:, 0], y=cluster_points[:, 1],
            mode='markers',
            marker=dict(size=6, color=colors[i % len(colors)], opacity=0.6),
            text=[f'Cluster {cluster}'] * len(cluster_points),
            hoverinfo='text',
            name=f'Cluster {cluster}'
        ))

    def add_ellipse(fig, points, color, n_std=2.0):
        if len(points) > 1:
            cov = np.cov(points.T)
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(xy=(np.mean(points[:, 0]), np.mean(points[:, 1])),
                          width=lambda_[0]*n_std*2, height=lambda_[1]*n_std*2,
                          angle=np.rad2deg(np.arccos(v[0, 0])),
                          edgecolor=color, fc='None', lw=2)

            # Generate the ellipse path
            ell_center = ell.get_center()
            ell_width = ell.width / 2
            ell_height = ell.height / 2
            angle = np.deg2rad(ell.angle)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            theta = np.linspace(0, 2 * np.pi, 100)
            x = ell_width * np.cos(theta)
            y = ell_height * np.sin(theta)

            x_rot = cos_angle * x - sin_angle * y + ell_center[0]
            y_rot = sin_angle * x + cos_angle * y + ell_center[1]

            path = f'M {x_rot[0]},{y_rot[0]} ' + ' '.join([f'L {x_},{y_}' for x_, y_ in zip(x_rot[1:], y_rot[1:])]) + ' Z'

            fig.add_shape(type="path",
                          path=path,
                          line=dict(color=color, width=2))

    # Add an ellipse for each cluster
    for i, cluster in enumerate(unique_clusters):
        cluster_points = train_pca[train_clusters == cluster]
        add_ellipse(fig, cluster_points, colors[i % len(colors)], n_std=2.2)

    # Add the new point after a delay
    if n > 1:
        fig.add_trace(go.Scatter(
            x=[data_pca[0, 0]], y=[data_pca[0, 1]],
            mode='markers',
            marker=dict(color='black', size=12),
            name='New Point'
        ))

    fig.update_layout(
        title='Cluster Visualization',
        xaxis_title='PCA 1',
        yaxis_title='PCA 2',
        legend=dict(
            x=1, y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)
