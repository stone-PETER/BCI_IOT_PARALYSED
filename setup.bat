@echo off
echo.
echo 🧠 BCI Real-time Classification System
echo =====================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo ✓ Python found
python --version

:: Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ Node.js not found. Please install Node.js 16+
    pause
    exit /b 1
)

echo ✓ Node.js found
node --version

:: Check if npm is installed
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ npm not found. Please install npm
    pause
    exit /b 1
)

echo ✓ npm found
npm --version

echo.
echo Checking model files...

:: Check for model files
if exist "CODE\bci_eegnet_model.h5" (
    echo ✓ Found: CODE\bci_eegnet_model.h5
) else (
    echo ⚠ Missing: CODE\bci_eegnet_model.h5
)

if exist "CODE\bci_preprocessor.pkl" (
    echo ✓ Found: CODE\bci_preprocessor.pkl
) else (
    echo ⚠ Missing: CODE\bci_preprocessor.pkl
)

if exist "CODE\bci_preprocessed_data.npz" (
    echo ✓ Found: CODE\bci_preprocessed_data.npz
) else (
    echo ⚠ Missing: CODE\bci_preprocessed_data.npz
)

echo.
echo Setting up backend...
cd backend

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ✗ Failed to install backend dependencies
    pause
    exit /b 1
)

echo ✓ Backend dependencies installed successfully

cd ..

echo.
echo Setting up frontend...
cd frontend

:: Install npm dependencies
echo Installing npm dependencies...
npm install

if %errorlevel% neq 0 (
    echo ✗ Failed to install frontend dependencies
    pause
    exit /b 1
)

echo ✓ Frontend dependencies installed successfully

cd ..

echo.
echo 🎉 Setup complete!
echo.
echo To start the system:
echo.
echo 1. Start the backend server:
echo    cd backend
echo    venv\Scripts\activate.bat
echo    python app.py
echo.
echo 2. In a new command prompt, start the frontend:
echo    cd frontend
echo    npm start
echo.
echo 3. Open your browser to:
echo    http://localhost:3000
echo.
echo For more information, see README.md
echo.
pause