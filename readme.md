# Hands-on Materials for IF4021

This repository contains hands-on materials for the IF4021 course at [Informatics Department](https://if.itera.ac.id), Institut Teknologi Suamtera ([ITERA](https://itera.ac.id)).

## Table of Contents
| No. | Topic            | Description | Notebook |
| --- | ---------------- | ----------- | -------- |
| 1   | Audio Processing |             |          |


## Visualization

This project contains interactive visualizations for various concepts in the IF4021 course. It's not mandatory for student to install this project on their side.

### Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed [Node.js](https://nodejs.org/) (version 12.0 or later) and npm (usually comes with Node.js).
* You have a Windows/Linux/Mac machine.

### Installing IF4021 Visualizations

To install the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/if4021-visualizations.git
   ```
2. Navigate to the project directory:
   ```
   cd if4021-visualizations
   ```
3. Install the dependencies:
   ```
   npm install
   ```

### Using IF4021 Visualizations

To use IF4021 Visualizations, follow these steps:

1. Start the development server:
   ```
   npm start
   ```
2. Open your web browser and navigate to `http://localhost:3000`

You should now see the visualization page with multiple interactive visualizations.

### Project Structure

The project is structured as follows:

```
react-visualizations/
├── public/
├── src/
│   ├── components/
│   │   └── visualizations/
│   │       ├── ADCVisualization.js
│   │       └── JPEGCompressionVisualization.js
│   │       └── and_others.js
│   ├── App.js
│   └── index.js
├── package.json
└── README.md
```

Each visualization is a separate component in the `src/components/visualizations/` directory.

## Building for Production

To create a production build, run:

```
npm run build
```

This will create a `build` directory with a production build of your app.

## Serving the Production Build

To serve the production build, you can use a static server:

1. Install `serve` globally (if not already installed):
   ```
   npm install -g serve
   ```
2. Serve your production build:
   ```
   serve -s build
   ```

The app will now be serving at `http://localhost:3000` (or another port if 3000 is busy).

## Contributing to IF4021 Visualizations

To contribute to IF4021 Visualizations, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).