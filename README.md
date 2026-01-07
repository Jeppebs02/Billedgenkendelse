# Billedgenkendelse

This is Gruppe 1's 4th semester project.

## Project Structure

*   **`App/KoerekortApp/`**: Frontend mobile application built with Vue.js, Ionic, and Capacitor.
*   **`Python-code/Billedegenkendelse/`**: Backend API and logic built with Python and Flask.

---

## 🐍 Backend Setup (API)

The backend API is containerized and designed to be run using Docker Compose.

### Prerequisites

*   [Docker](https://www.docker.com/get-started) and Docker Compose installed.

### Running the API

1.  **Network Setup**: The Docker configuration expects an external network named `npm_default`. Create it if it doesn't exist:
    ```bash
    docker network create npm_default
    ```
	
  ⚠️ **Important – Docker Network**
 
  The provided `docker-compose.yml` references an external Docker network named `npm_default` because this project originates from a production setup using Nginx Proxy Manager.
 
  **You are NOT required to use `npm_default`.**  
  You may attach the containers to **any Docker network you prefer**, as long as:
  - The network exists
  - The `docker-compose.yml` is updated accordingly
 
  For local development, you can either:
  - Create `npm_default`, **or**
  - Change the network name in `docker-compose.yml` to match your own setup


2.  **Start the Service**: Navigate to the `Python-code` directory and run:
    ```bash
    cd Python-code
    docker-compose up -d
    ```

The API will be accessible at `http://localhost:5001`.

### API Usage Note
The API requires an API key for access.
*   **Header**: `x-api-key`
*   **Default Key**: `a541fe33-6c48-490c-b71a-eadab16594de` (See `application.py`)

---

## 📱 Frontend Setup (App)

The frontend can be run in the browser or compiled to native mobile apps (Android/iOS) using Capacitor.

### Prerequisites

*   [Node.js](https://nodejs.org/)
*   [npm](https://www.npmjs.com/)
*   **For Mobile**: Android Studio (for Android) and Xcode (for iOS).


⚠️ **Important – API Endpoint Configuration**

This codebase originates from a production environment.  
The hostname **`api.terragrouplabs.net`** is hardcoded in the following files:

- `src/stores/dataStore.js` (line 21)
- `src/components/CameraComponent.vue` (line 64)

For local or self-hosted deployments, you MUST replace this hostname  
with the **IP address or hostname of your own API container**.

Example:
```js
http://192.168.1.100:5001
```

Failing to do this will cause the app to attempt requests against the original production API.

### Installation

Navigate to the application directory and install dependencies:

```bash
cd App/KoerekortApp
npm install
```

### Development (Browser)

To start the development server:

```bash
npm run dev
```
The app will typically run on `http://localhost:5173`.

### Mobile Compilation (Capacitor)

To compile and run the app on a physical device or emulator.

#### 🤖 Android

1.  **Build the web assets:**
    ```bash
    npm run build
    ```

2.  **Sync with native project:**
    ```bash
    npx cap sync android
    ```

3.  **Run on Device:**
    You can open the project in Android Studio to build and deploy:
    ```bash
    npx cap open android
    ```
    *Alternatively, if you have a device connected via ADB, you can try:*
    ```bash
    npx cap run android --target <DeviceID>
    ```
	
    *You can get your device id with:*
    ```bash
    adb devices
    ```

#### 🍎 iOS

1.  **Build the web assets:**
    ```bash
    npm run build
    ```

2.  **Sync with native project:**
    ```bash
    npx cap sync ios
    ```

3.  **Run on Device:**
    Open the project in Xcode and select your device:
    ```bash
    npx cap open ios
    ```
