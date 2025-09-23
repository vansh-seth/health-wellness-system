// components/Navbar.jsx
import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export const Navbar = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  if (!isAuthenticated) {
    return (
      <header className="header">
        <h1>Health & Wellness Tracker</h1>
        <div className="auth-links">
          {location.pathname !== '/login' && (
            <Link to="/login" className="auth-link">Login</Link>
          )}
          {location.pathname !== '/signup' && (
            <Link to="/signup" className="auth-link">Sign Up</Link>
          )}
        </div>
      </header>
    );
  }

  return (
    <>
      <header className="header">
        <h1>Health & Wellness Tracker</h1>
        <div className="user-info">
          <span>Welcome, {user?.first_name || 'User'}!</span>
          <button onClick={handleLogout} className="logout-btn">Logout</button>
        </div>
      </header>

      <nav className="nav-tabs">
        <Link 
          to="/dashboard" 
          className={location.pathname === '/dashboard' || location.pathname === '/' ? 'active' : ''}
        >
          Dashboard
        </Link>
        <Link 
          to="/log" 
          className={location.pathname === '/log' ? 'active' : ''}
        >
          Log Health Data
        </Link>
        <Link 
          to="/history" 
          className={location.pathname === '/history' ? 'active' : ''}
        >
          History
        </Link>
        <Link 
          to="/profile" 
          className={location.pathname === '/profile' ? 'active' : ''}
        >
          Profile
        </Link>
      </nav>
    </>
  );
};