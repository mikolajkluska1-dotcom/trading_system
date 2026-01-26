/**
 * Authentication API
 * Handles communication with the backend auth endpoints
 */

// Use Vite environment variable if available, otherwise default to localhost
// In production/Docker, VITE_API_URL should be set in docker-compose.yml
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Login user with username and password
 * @param {string} username 
 * @param {string} password 
 * @returns {Promise<Object>} User data with role
 */
export async function login(username, password) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
            credentials: 'include', // Include cookies for session management
        });

        if (!response.ok) {
            if (response.status === 401) {
                throw new Error('Invalid credentials');
            }
            throw new Error(`Login failed: ${response.statusText}`);
        }

        const data = await response.json();

        // Store user data in localStorage for persistence
        localStorage.setItem('redline_user', JSON.stringify(data));

        return data;
    } catch (error) {
        console.error('Login error:', error);
        throw error;
    }
}

/**
 * Logout current user
 */
export function logout() {
    // Clear user data from localStorage
    localStorage.removeItem('redline_user');

    // If backend has a logout endpoint, call it here
    // fetch(`${API_BASE_URL}/api/auth/logout`, { method: 'POST', credentials: 'include' });
}

/**
 * Get current user from localStorage
 * @returns {Promise<Object|null>} User data or null
 */
export async function getCurrentUser() {
    try {
        const userStr = localStorage.getItem('redline_user');
        if (!userStr) {
            return null;
        }

        const user = JSON.parse(userStr);

        // Optional: Verify with backend that session is still valid
        // const response = await fetch(`${API_BASE_URL}/api/auth/me`, { credentials: 'include' });
        // if (!response.ok) {
        //   logout();
        //   return null;
        // }

        return user;
    } catch (error) {
        console.error('Get current user error:', error);
        logout(); // Clear invalid data
        return null;
    }
}

/**
 * Register a new user (if registration is enabled)
 * @param {string} username 
 * @param {string} password 
 * @param {string} contact 
 * @returns {Promise<Object>} Registration result
 */
export async function register(username, password, contact) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password, contact }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Registration failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Registration error:', error);
        throw error;
    }
}
