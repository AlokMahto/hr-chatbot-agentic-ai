import os
import requests
from datetime import date, datetime
from typing import Dict, Any, Optional
from langchain.tools import Tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HOLIDAY_API_KEY = os.getenv("HOLIDAY_API_KEY")
HOLIDAY_API_BASE_URL = "https://calendarific.com/api/v2"

def get_current_date(query: str = "") -> str:
    """
    Get the current date in a formatted string.
    
    Args:
        query: Input query (not used, but required for LangChain Tool compatibility)
    
    Returns:
        str: Current date in format "YYYY-MM-DD (Day, Month DD, YYYY)"
    """
    today = date.today()
    return today.strftime("%Y-%m-%d (%A, %B %d, %Y)")

def check_holidays(query: str = "", country_code: str = "IN", year: Optional[int] = None) -> str:
    """
    Check holidays for a specific country and year using Calendarific API.
    
    Args:
        query: Input query (not used, but required for LangChain Tool compatibility)
        country_code: ISO 3166-1 alpha-2 country code (default: "IN" for India)
        year: Year to check holidays for (default: current year)
    
    Returns:
        str: Formatted string with holiday information or error message
    """
    if not HOLIDAY_API_KEY:
        return "Holiday API key is not configured. Please set HOLIDAY_API_KEY in environment variables."
    
    if year is None:
        year = date.today().year
    
    try:
        url = f"{HOLIDAY_API_BASE_URL}/holidays"
        params = {
            "api_key": HOLIDAY_API_KEY,
            "country": country_code,
            "year": year
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("meta", {}).get("code") != 200:
            return f"API Error: {data.get('meta', {}).get('error_detail', 'Unknown error')}"
        
        holidays = data.get("response", {}).get("holidays", [])
        
        if not holidays:
            return f"No holidays found for {country_code} in {year}."
        
        # Format the holidays
        result = f"Holidays in {country_code} for {year}:\n\n"
        for holiday in holidays[:20]:  # Limit to first 20 holidays
            name = holiday.get("name", "Unknown")
            date_info = holiday.get("date", {}).get("iso", "Unknown date")
            holiday_type = ", ".join(holiday.get("type", []))
            result += f"• {name} - {date_info} ({holiday_type})\n"
        
        if len(holidays) > 20:
            result += f"\n... and {len(holidays) - 20} more holidays."
        
        return result
        
    except requests.exceptions.Timeout:
        return "Holiday API request timed out. Please try again later."
    except requests.exceptions.RequestException as e:
        return f"Error fetching holidays: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def check_today_holiday(query: str = "", country_code: str = "IN") -> str:
    """
    Check if today is a holiday in the specified country.
    
    Args:
        query: Input query (not used, but required for LangChain Tool compatibility)
        country_code: ISO 3166-1 alpha-2 country code (default: "IN" for India)
    
    Returns:
        str: Information about today's holiday status
    """
    if not HOLIDAY_API_KEY:
        return "Holiday API key is not configured. Please set HOLIDAY_API_KEY in environment variables."
    
    today = date.today()
    
    try:
        url = f"{HOLIDAY_API_BASE_URL}/holidays"
        params = {
            "api_key": HOLIDAY_API_KEY,
            "country": country_code,
            "year": today.year,
            "month": today.month,
            "day": today.day
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        holidays = data.get("response", {}).get("holidays", [])
        
        if holidays:
            result = f"Yes! Today ({today.strftime('%B %d, %Y')}) is a holiday:\n\n"
            for holiday in holidays:
                name = holiday.get("name", "Unknown")
                holiday_type = ", ".join(holiday.get("type", []))
                description = holiday.get("description", "No description available")
                result += f"• {name} ({holiday_type})\n"
                if description and description != "No description available":
                    result += f"  Description: {description}\n"
            return result
        else:
            return f"No, today ({today.strftime('%B %d, %Y')}) is not a holiday in {country_code}."
            
    except requests.exceptions.RequestException as e:
        return f"Error checking today's holiday: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def get_upcoming_holidays(query: str = "", country_code: str = "IN", limit: int = 5) -> str:
    """
    Get upcoming holidays from today onwards.
    
    Args:
        query: Input query (not used, but required for LangChain Tool compatibility)
        country_code: ISO 3166-1 alpha-2 country code (default: "IN" for India)
        limit: Number of upcoming holidays to return (default: 5)
    
    Returns:
        str: Formatted string with upcoming holiday information
    """
    if not HOLIDAY_API_KEY:
        return "Holiday API key is not configured. Please set HOLIDAY_API_KEY in environment variables."
    
    today = date.today()
    
    try:
        url = f"{HOLIDAY_API_BASE_URL}/holidays"
        params = {
            "api_key": HOLIDAY_API_KEY,
            "country": country_code,
            "year": today.year
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        holidays = data.get("response", {}).get("holidays", [])
        
        # Filter upcoming holidays
        upcoming = []
        for holiday in holidays:
            holiday_date_str = holiday.get("date", {}).get("iso", "")
            if holiday_date_str:
                try:
                    # Handle ISO format with or without time component
                    # Split on 'T' to get just the date part
                    date_part = holiday_date_str.split('T')[0]
                    holiday_date = datetime.strptime(date_part, "%Y-%m-%d").date()
                    if holiday_date >= today:
                        upcoming.append(holiday)
                except ValueError:
                    # Skip holidays with invalid date formats
                    continue
        
        if not upcoming:
            return f"No upcoming holidays found for {country_code} in {today.year}."
        
        # Limit the results
        upcoming = upcoming[:limit]
        
        result = f"Upcoming holidays in {country_code}:\n\n"
        for holiday in upcoming:
            name = holiday.get("name", "Unknown")
            date_info = holiday.get("date", {}).get("iso", "Unknown date")
            holiday_type = ", ".join(holiday.get("type", []))
            result += f"• {name} - {date_info} ({holiday_type})\n"
        
        return result
        
    except Exception as e:
        return f"Error fetching upcoming holidays: {str(e)}"

# Create LangChain Tools
date_tool = Tool(
    name="get_current_date",
    func=get_current_date,
    description="Returns the current date. Use this when the user asks about today's date or what day it is."
)

holiday_checker_tool = Tool(
    name="check_holidays",
    func=check_holidays,
    description="Fetches all holidays for India in the current year. Use this when the user asks about holidays, public holidays, or leave calendar."
)

today_holiday_tool = Tool(
    name="check_today_holiday",
    func=check_today_holiday,
    description="Checks if today is a holiday in India. Use this when the user asks if today is a holiday or what holiday is today."
)

upcoming_holidays_tool = Tool(
    name="get_upcoming_holidays",
    func=get_upcoming_holidays,
    description="Gets the next 5 upcoming holidays in India. Use this when the user asks about upcoming holidays or next holidays."
)

# Function to be used in main.py for tool integration
def get_all_tools():
    """
    Returns all available tools for the agent.
    
    Returns:
        list: List of LangChain Tool objects
    """
    return [
        date_tool,
        holiday_checker_tool,
        today_holiday_tool,
        upcoming_holidays_tool
    ]