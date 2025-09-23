// services/api.js
const API_BASE = 'http://localhost:8000';

class ApiService {
  constructor() {
    this.baseURL = API_BASE;
  }

  getAuthHeaders() {
    const token = localStorage.getItem('token');
    return {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` })
    };
  }

  async handleResponse(response) {
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Network error' }));
      throw new Error(error.detail || 'Request failed');
    }
    return response.json();
  }

  // Auth endpoints
  async login(email, password) {
    const response = await fetch(`${this.baseURL}/auth/login`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify({ email, password })
    });
    return this.handleResponse(response);
  }

  async register(userData) {
    const response = await fetch(`${this.baseURL}/auth/register`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify(userData)
    });
    return this.handleResponse(response);
  }

  async getProfile() {
    const response = await fetch(`${this.baseURL}/profile`, {
      headers: this.getAuthHeaders()
    });
    return this.handleResponse(response);
  }

  async updateProfile(updates) {
    const response = await fetch(`${this.baseURL}/profile`, {
      method: 'PUT',
      headers: this.getAuthHeaders(),
      body: JSON.stringify(updates)
    });
    return this.handleResponse(response);
  }

  // Health logging endpoints
  async logMeal(mealData) {
    const response = await fetch(`${this.baseURL}/health/log/meal`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify(mealData)
    });
    return this.handleResponse(response);
  }

  async logExercise(exerciseData) {
    const response = await fetch(`${this.baseURL}/health/log/exercise`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify(exerciseData)
    });
    return this.handleResponse(response);
  }

  async logSleep(sleepData) {
    const response = await fetch(`${this.baseURL}/health/log/sleep`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify(sleepData)
    });
    return this.handleResponse(response);
  }

  async logMood(moodData) {
    const response = await fetch(`${this.baseURL}/health/log/mood`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify(moodData)
    });
    return this.handleResponse(response);
  }

  // Data retrieval endpoints
  async getHealthLogs(logType = null, days = 7) {
    const params = new URLSearchParams({ days });
    if (logType) params.append('log_type', logType);
    
    const response = await fetch(`${this.baseURL}/health/logs?${params}`, {
      headers: this.getAuthHeaders()
    });
    return this.handleResponse(response);
  }

  async getDashboardStats(days = 7) {
    const response = await fetch(`${this.baseURL}/dashboard/stats?days=${days}`, {
      headers: this.getAuthHeaders()
    });
    return this.handleResponse(response);
  }
}

export const api = new ApiService();